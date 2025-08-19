package main

import (
	"fmt"
	"math/bits"
	"unsafe"

	. "github.com/dmgard/guac"
)

func resample() {
	resample_f32_64_avx()
}

// resample_f32_64_avx generates a windowed sinc interpolator operating on sets of 8-sample registers
func resample_f32_64_avx() {
	fixed_resample_avx[float32, []float32](8, 8)
	fixed_resample_avx[float32, []float32](16, 16)
	_new_resample_f32_64_avx(8, 8)
	_new_resample_f32_64_avx(16, 16)
}

const fixedPointShift = 32

func tzcnt[I Integer](i I) int { return bits.TrailingZeros64(uint64(i)) }

type Integer interface {
	~int8 | ~int16 | ~int32 | ~int64 | ~int | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uint | ~uintptr
}

func fixed_resample_avx[T float32 | float64, S SliceTypes](simdVecLen, unrolls int) {
	var p struct {
		Out, In, Coefs  Reg[S]
		CoefIdx         Reg[int]
		OutIdx, OutStep Reg[int]
	}

	var r struct {
		CoefIdxOut Reg[int]
		OutIdxOut  Reg[int]
	}

	suffix := fmt.Sprintf("F%d_%dx%d", unsafe.Sizeof(*new(T))*8, simdVecLen, unrolls)

	Func("ResampleFixed"+suffix, NOSPLIT, &p, &r)
	// |--|--|--|--|--|--|--|--|--|--|--|--| // input samples
	// |------|------|------|------|------| // output blocks
	//       !     !     !        !     !   // input samples where an output block shift is triggered
	// TODO the same process is used to generate coefficients - the global sample offset must be saved and passed in between calls as well
	// instead of doing float conversion, map input samples/output samples to the range 0..MaxUint64. Add MaxUint64/(outRate*simdVecLen) * inRate each input sample to a subsample phase counter - when overflow occurs, output shift is needed.

	swizzle := func() []int {
		k := make([]int, unrolls-1)
		for i := range k {
			k[i] = i + 1
		}
		return k
	}()
	lowSwizzle := func() []int {
		k := make([]int, unrolls-1)
		for i := range k {
			k[i] = i
		}
		return k
	}()

	out := Iter[T](&p.Out, simdVecLen, unrolls)
	outShift := out.SwizzledUnrolls(swizzle...)
	in := Iter[T](&p.In)
	bcst := R[T](simdVecLen)
	coefs := Iter[T](&p.Coefs, simdVecLen, unrolls)
	coefIdx := p.CoefIdx.Init().Load()
	coefsLen := Len[T, int](coefs).Init().Load()
	outIdx := p.OutIdx.Init().Load()

	// TODO this throws bad operands
	outLenMask := Len[T, int](out).Init().Load().Sub(int32(1))

	outStep := p.OutStep.Init().Load()
	Comment("Reload previous partially accumulated output samples")

	out.Load()

	outAlignedIdx := outIdx.CloneDef()

	taps := simdVecLen * unrolls

	RangeOver(in, func(i *Reg[int]) {
		Comment("Compute left output sample index as fixedPointIndex / fixedPointScale - taps/2")
		outMin := outIdx.CloneDef().BitRshift(outIdx, int8(fixedPointShift)).
			Sub(int32(taps / 2))

		Comment("Compute output vector index as ",
			"(fixedPointIndex / fixedPointScale / vectorLength * vectorLength) % outBufferLength",
			"wraps within the output buffer and quantizes the nearest vector register multiple")
		lg2vecLn := int8(tzcnt(simdVecLen))
		outIdxToOutVecShift := fixedPointShift + lg2vecLn
		// TODO somewhat redudnant to do this every time when it only shifts when
		// the later CMOV test succeeds
		outAlignedIdx.BitRshift(outIdx, outIdxToOutVecShift).
			BitLshift(lg2vecLn).
			And(outLenMask)
		SetIndex(outAlignedIdx, out)

		// TODO why
		Comment("The coefficient load index is (filtertaps+padding) * wrappedInputIndex",
			"+ (outAlignedIdx*vectorLength - outSampleIdx")
		Comment("This loads each coefficient set within a block with proper zero padding",
			"so that multiple coefficient sets can be accumulated into one set of registers,",
			"offset by the proper number of in-register samples")
		SetIndex(coefIdx.Copy().Add(outAlignedIdx).Sub(outMin), coefs)

		Comment("Broadcast the current input sample and contribute and accumulate its output-phase-specific-coefficient-scaled individual contribution to every output sample in range")
		bcst.Broadcast(in.Addr())
		out.AddProductOf(
			bcst.BroadcastUnrolled(unrolls),
			coefs.Addr(),
		)

		Comment("If incrementing the output index crosses a multiple of vectorLength,",
			"the lowest register is completely accumulated and can be stored while the rest",
			"are shifted down in its place")
		// TODO could probably do this with a bit test but const shifts might actually be faster
		outIdx.Copy().BitRshift(outIdxToOutVecShift).Compare(
			outIdx.Copy().Add(outStep).BitRshift(outIdxToOutVecShift)).
			JumpE("no_store")
		{
			out.SwizzledUnrolls(0).Store()
			outShift.Store(out.SwizzledUnrolls(lowSwizzle...))
			out.SwizzledUnrolls(unrolls - 1).Xor()
			outIdx.Add(int32(simdVecLen))
		}
		Label("no_store")

		Comment("Update and wrap coefficient index")
		phaseScratch := R[int]().Xor()
		coefIdx.Add(int32(taps + 2*simdVecLen)).Compare(coefsLen)
		Comment("Wrap phase counter - SUB changes flags so do this after to avoid clobbering Compare result")
		coefIdx.Sub(phaseScratch.MoveIf_GE(coefsLen))
	}, out)

	Comment("Store each partially accumulated vector to the output slice")
	Comment("taking care to wrap into output ringbuffer")
	for i := range unrolls {
		// TODO int constant in Add generates "bad operands" instead automatic convert or
		// type error
		out := out.ByteOffsetAllTo(0)
		SetIndex(outAlignedIdx.Add(int32(simdVecLen)).And(outLenMask),
			out.SwizzledUnrolls(i).Store())
	}

	ZeroUpper()
	Comment("Return the latest phase and output index for reuse in future calls")
	r.CoefIdxOut.Init().Addr().Load(coefIdx)
	r.OutIdxOut.Init().Addr().Load(outIdx)

	Ret()
}

func _new_resample_f32_64_avx(simdVecLen, unrolls int) {
	var p struct {
		Out, In, Coefs               Reg[[]float32]
		PhaseIdx, Phases             Reg[int]
		SubsampleIdx, SubsampleDelta Reg[uint64]
		Taps                         Reg[int]
	}

	var r struct {
		PhaseIdxOut     Reg[int]
		SubsampleIdxOut Reg[uint64]
	}

	suffix := fmt.Sprintf("_%dx%d", simdVecLen, unrolls)

	Func("ResampleF32x64"+suffix, NOSPLIT, &p, &r)
	// |--|--|--|--|--|--|--|--|--|--|--|--| // input samples
	// |------|------|------|------|------| // output blocks
	//       !     !     !        !     !   // input samples where an output block shift is triggered
	// TODO the same process is used to generate coefficients - the global sample offset must be saved and passed in between calls as well
	// instead of doing float conversion, map input samples/output samples to the range 0..MaxUint64. Add MaxUint64/(outRate*simdVecLen) * inRate each input sample to a subsample phase counter - when overflow occurs, output shift is needed.

	swizzle := func() []int {
		k := make([]int, unrolls-1)
		for i := range k {
			k[i] = i + 1
		}
		return k
	}()
	lowSwizzle := func() []int {
		k := make([]int, unrolls-1)
		for i := range k {
			k[i] = i
		}
		return k
	}()

	out := Iter[float32](&p.Out, simdVecLen, unrolls)
	outShift := out.SwizzledUnrolls(swizzle...)
	in := Iter[float32](&p.In)
	bcst := R[float32](simdVecLen)
	coefs := Iter[float32](&p.Coefs, simdVecLen, unrolls)
	phaseIdx := p.PhaseIdx.Init().Load()
	phases := p.Phases.Init().Load()
	subIdx := p.SubsampleIdx.Init().Load()
	subDelta := p.SubsampleDelta.Init().Load()
	taps := p.Taps.Init().Load()
	Comment("Reload previous partially accumulated output samples")
	out.Load()

	outIdx := R[int]().Xor()

	RangeOver(in, func(i *Reg[int]) {
		SetIndex(outIdx, out)
		Comment("The coefficient load index is filtertaps * phase index")
		coefIdx := phaseIdx.Copy().Mul(taps)
		//coefIdx := phaseIdx.Copy().BitLshift(int8(tapsLog2))
		SetIndex(coefIdx, coefs)

		Comment("Broadcast the current input sample and contribute and accumulate its output-phase-specific-coefficient-scaled individual contribution to every output sample in range")
		bcst.Broadcast(in.Addr())
		out.AddProductOf(
			bcst.SwizzledUnrolls(make([]int, unrolls)...), // Broadcast the 0 register to all unrolls
			coefs.Addr(),
		)

		Comment("If adding maxUint/resampleRatio/simdLen overflows the subsample counter, advance to the next output block",
			"The lowest block is completed and further input samples have no contribution")
		subIdx.Add(subDelta).JumpNO("no_store")
		{
			out.SwizzledUnrolls(0).Store()
			outShift.Store(out.SwizzledUnrolls(lowSwizzle...))
			out.SwizzledUnrolls(unrolls - 1).Xor()
			outIdx.Add(int32(simdVecLen))
		}
		Label("no_store")

		Comment("Update and wrap phase index counter")
		phaseScratch := R[int]().Xor()
		phaseIdx.Add(int32(1)).Compare(phases)
		Comment("If phase index counter was wrapped, reset subsample counter to one")
		subIdx.MoveIf_GE(R[uint64]().Load(uint64(1)))
		Comment("Wrap phase counter - SUB changes flags so do this after to avoid clobbering Compare result")
		phaseIdx.Sub(phaseScratch.MoveIf_GE(phases))
	}, out)

	Comment("Save partial outputs")
	out.Store()

	ZeroUpper()
	// TODO need to return exact input sample and output block positions as well
	Comment("Return the latest phase index for reuse in future calls")
	r.PhaseIdxOut.Init().Addr().Load(phaseIdx)
	r.SubsampleIdxOut.Init().Addr().Load(subIdx)

	Ret()
}

func gen_resample_coefs_f32_64_avx() {
	// given window, tap count, resample ratio, phase count, coefs array storage
	// for every sample
	// for every element in each output register:
	// compute input-space floating point fractional sample index of output sample center
	// compute distance from current input sample in floating point fractional input samples
	// evaluate windowed sinc at each calculated distance: these are the coefficients
	// end for every element
	// write out coefficient block
	// compute wrapped phase idx - if phase shifts between output blocks, add len(outputBlock) fractional input-space samples to the base input sample center: this accounts for output block-aligned sample drift due to resampling rate
	// end for every sample

	const (
		simdVecLen = 8
		unrolls    = 8
		tapsLog2   = 6 // TODO need to replace with IMUL to allow non-power-of-two filter tap lengths
	)

	var p struct {
		Coefs          Reg[[]float32]
		Phases, Taps   Reg[int]
		SubsampleDelta Reg[uint64]
	}

	Func("GenResampleCoefsF32x64Avx", NOSPLIT, &p)
	// |--|--|--|--|--|--|--|--|--|--|--|--| // input samples
	// |------|------|------|------|------| // output blocks
	//       !     !     !        !     !   // input samples where an output block shift is triggered
	// TODO the same process is used to generate coefficients - the global sample offset must be saved and passed in between calls as well
	// instead of doing float conversion, map input samples/output samples to the range 0..MaxUint64. Add MaxUint64/(outRate*simdVecLen) * inRate each input sample to a subsample phase counter - when overflow occurs, output shift is needed.

	coefs := Iter[float32](&p.Coefs, simdVecLen, unrolls)
	phases := p.Phases.Init().Load()
	subDelta := p.SubsampleDelta.Init().Load()
	subIdx := R[int]().Xor()

	// for each sample
	i := R[int]().Xor()
	Label("loopStart")
	{
		Comment("The coefficient write index is filtertaps * phase index")
		coefIdx := i.Copy().BitLshift(int8(tapsLog2))
		SetIndex(coefIdx, coefs)
		// load 0..simdVecLen into register
		// scale into input space using resampling ratio: these are the input-space centerpoints of the sinc filters centered at each output sample, relative to the current input sample TODO this could be precomputed: all phases have the same initial input-space output centers
		// compute inputSpaceCenters - (inputSpaceSample-outputBlockStartInInputSpace)
		// the output block start only moves on a block shift, which may take many input samples
		// scale into -1:1 scaled range of windowed sinc TODO combine this with initial resampling scale and apply it to the inputSpaceSample and outputBlockStart locations
		// compute windowed sinc of one output coef block at a time using approx sin(pi*centerRelativeSample)/(pi*centerRelativeSample) * windowFunc(centerRelativeSample), for all coef blocks (in a loop, instead of in registers: approx sinc needs nearly all registers)
		// advance through all phases

		Comment("If adding maxUint/resampleRatio/simdLen overflows the subsample counter, advance to the next output block",
			"The lowest block is completed and further input samples have no contribution")
		subIdx.Add(subDelta).JumpNO("no_store")
		{
			// TODO here is where we conditionally add to the base offset in... output time? whichever it is that shifts the filter kernels based on the offset of the output registers the coefficients are being extracted to
		}
		Label("no_store")
	}
	i.Add(int32(1)).Compare(phases).JumpLE("loopStart")

	ZeroUpper()
	Ret()
}
