package resample

import (
	"math"
	"unsafe"

	. "github.com/klauspost/cpuid/v2"
)

type SimdResampler[T Sample] struct {
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

// NewSIMD constructs a resampler with precomputed sinc coefficients
// TODO will be moved into regular resampler and used automatically based on
// CPUID support
func NewSIMD[T Sample, S Scalar](_srIn, _srOut S, taps int) (s *SimdResampler[T]) {
	s = &SimdResampler[T]{consts: new(consts[T])}

	ratio := Ffdiv(_srIn, _srOut)

	srIn, srOut := int(_srIn), int(_srOut)

	taps = CeiledDivide(taps, 2) * 2 // round taps to multiple of 2
	// TODO for SIMD pad to multiple of vector length

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
		s.taps = RoundUpMultPow2(2*s.delay+1, 16)
		s.coefs = make([]T, s.taps*phases)

		// TODO temporary for testing
		for i := range s.coefs {
			s.coefs[i] = 1
		}

		return
		var outIdx fixed64
		// one unique set of filter taps per reduced output sample rate index
		for i := range phases {
			outPos := float64(outIdx) / float64(fixedPointOne)
			//outPos := float64(outIdx >> fixedPointShift)

			ci := i * s.taps
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
	nil,
	nil,
	sliceOf(nil,
		ResampleFixedF32_8x2, ResampleFixedF32_8x3, ResampleFixedF32_8x4,
		ResampleFixedF32_8x5, ResampleFixedF32_8x6, ResampleFixedF32_8x7,
		ResampleFixedF32_8x8),
	sliceOf(nil,
		ResampleFixedF32_16x2, ResampleFixedF32_16x3, ResampleFixedF32_16x4,
		ResampleFixedF32_16x5, ResampleFixedF32_16x6, ResampleFixedF32_16x7,
		ResampleFixedF32_16x8, ResampleFixedF32_16x9, ResampleFixedF32_16x10,
		ResampleFixedF32_16x11, ResampleFixedF32_16x12, ResampleFixedF32_16x13,
		ResampleFixedF32_16x14, ResampleFixedF32_16x15, ResampleFixedF32_16x16),
)
var simdLevel = func() int {
	switch {
	case CPU.Supports(AVX, AVX512DQ, AVX512F, CMOV):

		return 4
	case CPU.Supports(AVX, CMOV, FMA3):
		return 3
	case CPU.Supports(SSE2, CMOV): // TODO
		return 2
	}

	return 1
}()

func (s *SimdResampler[T]) Process(in []T) {
	// TODO chunk input to avoid overflowing output buffer

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
		panic("TODO")
	}

	// IFF this is a rational-approximation resampler,
	// accumulate drift relative to ideal sample rate
	// swap to alternate undershoot/overshoot resampler if clock drift is too high
	if s.driftStep != 0 && s.coefsIdx < coefIn {
		s.drift += s.driftStep
		if Sign(s.drift) == Sign(s.driftStep) { // TODO variable threshold?
			s.consts, s.alt = s.alt, s.consts
		}
	}
}

func (s *SimdResampler[T]) Read(into []T) int {
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
