package resample

import (
	"math"
	"unsafe"

	. "github.com/klauspost/cpuid/v2"
)

type Resampler[T Sample] struct {
	out []T

	// output time in fixed-point ratio of input samples
	outIdx fixed64
	// last read output sample location
	readIdx int
	// input sample count for selecting output phases
	coefsIdx int
	// accumulated output-time drift due to rational sample ratio approximation
	drift float64

	*consts[T]
	alt *consts[T]
}

type fixed64 int64

const fixedPointShift = 32
const fixedPointOne = 1 << fixedPointShift
const maxPhases = 512

type Sample interface {
	float32 | float64 | int8 | int16 | int32 | int64
}
type consts[T Sample] struct {
	// ratio of output samples traversed per input sample
	outStep fixed64

	ratio, invRatio,
	// TODO explain
	scaleFactor, sincFactor float64

	// half of the filter width in output samples fixed-point time
	halfTapsFixedPoint fixed64
	// half filter width in output sample indices, and the full width
	delay, taps int
	// used to window output time for sinc coefficient generation
	// TODO doesn't need to be kept after initialization does it?
	invFilterWidth float64

	// precomputed sinc coefficients
	coefs []T

	// output clock drift per input sample
	driftStep float64
}

// New constructs a resampler with precomputed sinc coefficients
func New[T Sample, S Scalar](_srIn, _srOut S, taps int) (s *Resampler[T]) {
	s = &Resampler[T]{consts: new(consts[T])}

	ratio := Ffdiv(_srIn, _srOut)

	srIn, srOut := int(_srIn), int(_srOut)

	// SIMD pad to multiple of vector length
	vecLen := 1 << simdLevel
	if simdLevel <= slSSE {
		vecLen = 0
	} else {
		taps = RoundUpMultPow2(taps, vecLen) // may as well use the entire register
	}
	// TODO use fallback if too many taps requested
	//taps = min(taps, simdMaxFiltLens[simdLevel])

	initConsts := func() {
		s.ratio = ratio
		s.invRatio = 1 / ratio

		// advance output fixed point time this much per new input sample
		s.outStep = fixed64(math.Ceil(s.invRatio * fixedPointOne))
		// downsampling accumulates ratio input samples per output sample
		// rescale output to maintain unity gain
		s.scaleFactor = min(s.invRatio, 1)
		// when upsampling, scale sinc inputs to low-pass at lower input rate
		// avoids aliasing
		s.sincFactor = min(s.ratio, 1)

		// output is a power of two ringbuffer, padded and scaled up for upsampling many more
		// output samples than were input
		s.out = make([]T,
			RoundUpPow2(
				FmulCeiled(
					taps*4,
					max(s.invRatio, 1)),
			),
		)

		// convolved samples need to be output half the filter length ahead
		// this way they accumulate just in time to be read with a minimal delay
		s.delay = CeiledDivide(taps, 2)

		// compute the fixed point distance of the filter offset for computing sinc's window
		// window functions are from 0 ... 1, sinc is -taps/2 ... taps/2
		s.halfTapsFixedPoint = CeiledDivide(fixed64(taps<<fixedPointShift), 2)
		// scale the sinc window both by the width of the filter and the resample ratio
		// if and only if the resample ratio is <1, i.e. upsampling, always low-passing at the
		// lower sample rate to avoid aliasing
		s.invFilterWidth = 1 / float64(taps) / s.sincFactor

		phases := srIn

		// precompute coefficients for each unique phase of the windowed sinc filter
		// round to nearest SIMD vector width
		// TODO bespoke routines for very small filter lengths?
		s.taps = taps
		// and pad by two SIMD registers
		paddedTaps := s.taps + 2*vecLen
		s.coefs = make([]T, paddedTaps*phases)

		var outIdx fixed64
		// one unique set of filter taps per reduced output sample rate index
		for i := range phases {
			outPos := float64(outIdx) / float64(fixedPointOne)
			//outPos := float64(outIdx >> fixedPointShift)

			// TODO shouldn't this just be offset to the proper zero padding directly?
			// TODO why calculate that in assembly?
			// deposit as |padding|coefficients|padding|
			ci := i*paddedTaps + vecLen
			for fi := range s.coefs[ci:][:s.taps] {
				// center a sinc on each outPos within the filter spread and compute coefficients
				coef := T(windowedSinc(
					(outPos-float64(fi-s.delay))*s.sincFactor,
					float64(s.delay)*s.sincFactor,
					s.invFilterWidth),
				)
				s.coefs[ci+fi] = coef * T(s.scaleFactor)
			}
			outIdx += s.outStep
			outIdx &= fixedPointOne - 1 // only care about the fractional part
		}
	}

	// too many phases, quantize and approximate
	if srIn > maxPhases {
		idealOutPerIn := 1 / ratio
		srInAlt, srOutAlt := srIn, srOut

		// TODO limit denominator by maximum desired/expected error?
		// i.e. prevent unlimited sample drift from accumulating at high ratios
		srIn, srOut, srInAlt, srOutAlt = FareySearch(ratio, maxPhases, 1000000)
		ratio = Ffdiv(srIn, srOut)
		// track sample drift from ideal and switch+interpolate between over/undershoot
		initConsts()
		// how many more output samples are being generated per input sample than
		// would be expected at the true ratio
		// premultiply by number of input samples per phase reset
		s.consts.driftStep = F64mul(s.consts.invRatio-idealOutPerIn, srIn)

		// if there is drift TODO or external drift sync
		// use the second-best rational approximation to compute compensating slower resample
		if s.consts.driftStep != 0 {
			s.alt = new(consts[T])
			s.alt, s.consts = s.consts, s.alt

			// generate second resampler and second coefficients for lower quantized ratio
			srIn, srOut, srInAlt, srOutAlt = srInAlt, srOutAlt, srIn, srOut
			ratio = Ffdiv(srIn, srOut)
			initConsts()
			s.alt.driftStep = F64mul(s.alt.invRatio-idealOutPerIn, srInAlt)
		}

		return s
	} else {
		initConsts()
		s.alt = s.consts
	}

	return s
}

var resampleFuncsF32 = sliceOf(
	nil,
	nil, // Scalar
	nil, // SSE TODO
	sliceOf(nil,
		ResampleFixedF32_8x2, ResampleFixedF32_8x3, ResampleFixedF32_8x4,
		ResampleFixedF32_8x5, ResampleFixedF32_8x6, ResampleFixedF32_8x7,
		ResampleFixedF32_8x8, ResampleFixedF32_8x9, ResampleFixedF32_8x10,
		ResampleFixedF32_8x11, ResampleFixedF32_8x12, ResampleFixedF32_8x13,
		ResampleFixedF32_8x14, ResampleFixedF32_8x15),
	sliceOf(nil,
		ResampleFixedF32_16x2, ResampleFixedF32_16x3, ResampleFixedF32_16x4,
		ResampleFixedF32_16x5, ResampleFixedF32_16x6, ResampleFixedF32_16x7,
		ResampleFixedF32_16x8, ResampleFixedF32_16x9, ResampleFixedF32_16x10,
		ResampleFixedF32_16x11, ResampleFixedF32_16x12, ResampleFixedF32_16x13,
		ResampleFixedF32_16x14, ResampleFixedF32_16x15, ResampleFixedF32_16x16,
		ResampleFixedF32_16x17, ResampleFixedF32_16x18, ResampleFixedF32_16x19,
		ResampleFixedF32_16x20, ResampleFixedF32_16x21, ResampleFixedF32_16x22,
		ResampleFixedF32_16x23, ResampleFixedF32_16x24, ResampleFixedF32_16x25,
		ResampleFixedF32_16x26, ResampleFixedF32_16x27, ResampleFixedF32_16x28,
		ResampleFixedF32_16x29, ResampleFixedF32_16x30, ResampleFixedF32_16x31,
	),
)

const (
	slInvalid = iota
	slScalar
	slSSE
	slAVX
	sl512
)

var simdLevel = func() int {
	switch {
	case CPU.Supports(AVX, AVX512DQ, AVX512F, CMOV):

		return sl512
	case CPU.Supports(AVX, CMOV, FMA3):
		return slAVX
	case CPU.Supports(SSE2, CMOV): // TODO
		return slSSE
	}

	return slScalar
}()

var simdMaxFiltLens = []int{
	slInvalid: math.MaxInt,
	slScalar:  math.MaxInt,
	slSSE:     math.MaxInt,
	slAVX:     8*16 - 8*2,
	sl512:     16*32 - 16*2,
}

func (s *Resampler[T]) Process(in []T) {
	// TODO chunk input to avoid overflowing output buffer
	// TODO and coefficient slice

	// scalar fallback
	if simdLevel < slSSE || s.taps > simdMaxFiltLens[simdLevel] {
		// TODO fallback to unlimited length SIMD filter routines
		s.processScalar(in)
		return
	}

	// TODO kludge to detect when phase wrap occurred
	coefIn := s.coefsIdx

	switch unsafe.Sizeof(*new(T)) * 8 {
	case 32:
		fn := resampleFuncsF32[simdLevel][s.taps>>simdLevel]

		coefIdx, outIdx := fn(
			SliceCast[float32](s.out),
			SliceCast[float32](in),
			SliceCast[float32](s.coefs), s.coefsIdx, int(s.outIdx), int(s.outStep))
		s.coefsIdx = coefIdx
		s.outIdx = fixed64(outIdx)
	case 64:
		fallthrough
	default:
		panic("TODO")
	}

	// IFF this is a rational-approximation resampler,
	// accumulate drift relative to ideal sample rate
	// swap to alternate undershoot/overshoot resampler if clock drift is too high
	if s.driftStep != 0 && s.coefsIdx < coefIn {
		s.drift += s.driftStep
		if Sign(s.drift) == Sign(s.driftStep) { // TODO variable threshold?
			s.consts, s.alt = s.alt, s.consts
			s.coefsIdx = 0
		}
	}
}

func (s *Resampler[T]) processScalar(in []T) {
	for _, input := range in {
		// weight contribution of this input sample to a patch the size of the filter
		// and accumulate to output samples at integer output slice indices
		// TODO might be off by one relative to floating point calculation
		outMin := int(s.outIdx>>fixedPointShift) - s.delay

		s.outIdx += s.outStep // + 1

		// coefs contains precomputed centered windowed sinc on each output sample
		for range s.taps {
			// wrap into output buffer
			// delay output by half the filter taps so all inputs can accumulate in time
			s.out[(outMin+s.delay)&(len(s.out)-1)] += input * s.coefs[s.coefsIdx]
			s.coefsIdx++
			outMin++
		}

		// wrap sample coefficients index
		if s.coefsIdx >= len(s.coefs) {
			s.coefsIdx = 0

			// IFF this is a rational-approximation resampler,
			// accumulate drift relative to ideal sample rate
			// swap to alternate undershoot/overshoot resampler if clock drift is too high
			s.drift += s.driftStep
			if Sign(s.drift) == Sign(s.driftStep) { // TODO variable threshold?
				s.consts, s.alt = s.alt, s.consts
			}
		}
	}
}

func (s *Resampler[T]) Read(into []T) int {
	n := len(into)
	ln := min(int(s.outIdx>>fixedPointShift)-s.readIdx, n)
	nextOutIdx := s.readIdx + ln
	wrapped := s.readIdx & (len(s.out) - 1)
	end := nextOutIdx & (len(s.out) - 1)

	if end >= wrapped {
		chunk := s.out[wrapped:end]
		into = into[copy(into, chunk):]
		_memClr(chunk)
	} else {
		into = into[copy(into, s.out[wrapped:]):]
		into = into[copy(into, s.out[:end]):]
		_memClr(s.out[wrapped:])
		_memClr(s.out[:end])
	}

	s.readIdx = nextOutIdx

	return ln
}

func FareySearch[T Scalar, F Float](target F, maxNum, maxDenom T) (num, denom, numAlt, denomAlt T) {
	if target > 1 {
		denom, num, denomAlt, numAlt = _fareySearch(1/target, maxDenom, maxNum)
	} else {
		num, denom, numAlt, denomAlt = _fareySearch(target, maxNum, maxDenom)
	}

	f64t := float64(target)
	// return closest ratio first
	if Abs(Ffdiv(num, denom)-f64t) < Abs(Ffdiv(numAlt, denomAlt)-f64t) {
		return numAlt, denomAlt, num, denom
	}
	return
}

func _fareySearch[T Scalar, F Float](target F, maxNum, maxDenom T) (num, denom, numAlt, denomAlt T) {
	numLo, denomLo := T(0), T(1)
	numHi, denomHi := T(1), T(1)

	for {
		num, denom = numLo+numHi, denomLo+denomHi

		if num > maxNum || denom > maxDenom {
			if target-Ftdiv[F](numLo, denomLo) < Ftdiv[F](numHi, denomHi)-target {
				return numLo, denomLo, numHi, denomHi
			}
			return numHi, denomHi, numLo, denomLo
		}

		if Ftdiv[F](num, denom) < target {
			numLo, denomLo = num, denom
		} else {
			numHi, denomHi = num, denom
		}
	}
}

func windowedSinc[T Float](x, widthHalf, widthInverse T) T {
	sinc := normalizedSinc(x)

	// blackman-harris window
	x -= widthHalf
	x *= widthInverse
	windowed := blackmanHarris(-x)
	return windowed * sinc // sinc
}

func normalizedSinc[T Float](x T) T {
	if Approx(0.000001, x, 0.0) {
		return 1
	}

	in := float64(x)
	return T(math.Sin(math.Pi*in) / (math.Pi * in))
}

func blackmanHarris[T Scalar](_in T) T {
	in := float64(_in)

	windowed := 0.35875 -
		0.48829*math.Cos(2*math.Pi*in) +
		0.14128*math.Cos(4*math.Pi*in) -
		0.01168*math.Cos(6*math.Pi*in)
	return T(windowed)
}
