package resample

import (
	"math"
	"math/big"
)

type fixed64 int64

const fixedPointShift = 32
const fixedPointOne = 1 << fixedPointShift

type Sample interface {
	float32 | float64 | int8 | int16 | int32 | int64
}

type OfflineSincResampler[T Sample] struct {
	out []T

	// output time in fixed-point ratio of input samples
	outIdx fixed64
	// last processed full chunk of quantum samples
	quantumIdx int
	// input sample count for selecting output phases
	coefsIdx int
	// accumulated output-time drift due to rational sample ratio approximation
	drift float64

	*consts[T]
	alt *consts[T]
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

	// length of samples to accumulate before reading/advancing an output chunk
	quantum    int
	logQuantum int

	// precomputed sinc coefficients
	coefs []T

	// output clock drift per input sample
	driftStep float64
}

// New constructs a resampler with precomputed sinc coefficients
func New[T Sample](srIn, srOut, quantum, taps int) (s *OfflineSincResampler[T]) {
	if quantum != RoundUpPow2(quantum) {
		panic("quantum must be a power of 2")
	}
	s = &OfflineSincResampler[T]{consts: new(consts[T])}

	inR, outR := big.NewInt(int64(srIn)), big.NewInt(int64(srOut))
	gcd := big.NewInt(0).GCD(nil, nil, inR, outR)

	srIn /= int(gcd.Int64())
	srOut /= int(gcd.Int64())

	ratio := Ffdiv(srIn, srOut)

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

		// TODO does taps need to scale up when sinc is widened due to resample ratio?
		//taps = Fmul(taps, s.ratio)

		// output is a power of two ringbuffer, padded
		s.out = make([]T, RoundUpPow2(max(taps, quantum)*4))

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

		// output chunk size. Will accumulate this many samples before pushing to the next node
		s.quantum = quantum
		s.logQuantum = tzcnt(quantum)

		phases := srIn

		// precompute coefficients for each unique phase of the windowed sinc filter
		s.taps = 2*s.delay + 1
		s.coefs = make([]T, s.taps*phases)

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
	const maxPhases = 512 // TODO adjust to limit maximum sample drift
	if srIn > maxPhases {
		idealInPerOut := ratio
		quantScale := Ffdiv(maxPhases, srIn)
		srIn = Fmul(srIn, quantScale)
		srOut = Fmul(srOut, quantScale)

		ratio = Ffdiv(srIn, srOut)

		// generate second resampler and second coefficients for lower or higher quantized ratio
		// track sample drift from ideal and switch+interpolate between over/undershoot
		initConsts()
		s.alt, s.consts = s.consts, new(consts[T])
		srInAlt := srIn

		// TODO adjust ratios to minimize drift discrepancy
		if ratio > idealInPerOut { // old ratio was lower, alternate will be lower
			srIn = Fmul(srOut, idealInPerOut)
		} else { // old ratio was higher, approximate it with overshoot
			srIn = FmulCeiled(srOut, idealInPerOut)
		}

		ratio = Ffdiv(srIn, srOut)
		initConsts()

		idealOutPerIn := 1 / idealInPerOut

		// how many more/fewer output samples are being generated per input sample than
		// would be expected at the true ratio
		// premultiply by number of input samples per phase reset
		s.consts.driftStep, s.alt.driftStep =
			F64mul(s.consts.invRatio-idealOutPerIn, srIn),
			F64mul(s.alt.invRatio-idealOutPerIn, srInAlt)
	} else {
		initConsts()
		s.alt = s.consts
	}

	return s
}

func (s *OfflineSincResampler[T]) Process(in []T) {
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

func (s *OfflineSincResampler[T]) Read(into []T) int {
	n := len(into)
	for nextOutChunkIdx := int(s.outIdx >> s.logQuantum >> fixedPointShift); s.quantumIdx < nextOutChunkIdx && len(into) >= s.quantum; s.quantumIdx++ {
		wrapped := (s.quantumIdx << s.logQuantum) & (len(s.out) - 1)
		chunk := s.out[wrapped:][:s.quantum]
		into = into[copy(into, chunk):]
		_memClr(chunk)
	}
	return n - len(into)
}

func (s *OfflineSincResampler[T]) putCoefs() {
	looped := false
	for {
		// convert fixed point output sample time to floating point approximation
		outPos := float64(s.outIdx) / float64(fixedPointOne)
		// TODO is filter length in input samples or output samples?
		// accumulate contribution of this input sample to a patch the size of the filter
		// and deposit output samples at integer slice indices
		outMin, outMax := int((s.outIdx-s.halfTapsFixedPoint)>>fixedPointShift), int((s.outIdx+s.halfTapsFixedPoint)>>fixedPointShift)

		// TODO wrap output index to avoid float precision issues at very high counts
		s.outIdx += s.outStep

		// center a windowed sinc on each output sample and calculate the weighted
		// contribution of the current input sample against that sinc window
		for oi := outMin; oi <= outMax; oi++ {
			coef := T(windowedSinc(
				(outPos-float64(oi))*s.sincFactor,
				float64(s.delay)*s.sincFactor,
				s.invFilterWidth),
			)

			s.coefs[s.coefsIdx+oi-outMin] = coef * T(s.scaleFactor)
		}

		// increment sample coefficients index, wrap
		if s.coefsIdx += s.taps; s.coefsIdx >= len(s.coefs) {
			s.coefsIdx = 0
			if looped {
				return
			}
			looped = true
		}
	}
}

type IntegerTimedSincResampler[T Sample] struct {
	out []T

	// output time in fixed-point ratio of input samples
	outIdx  fixed64
	outStep fixed64

	ratio, invRatio,
	scaleFactor, sincFactor float64
	halfTaps       fixed64
	delay          int
	invFilterWidth float64

	quantum    int
	logQuantum int

	// last processed full chunk of quantum samples
	quantumIdx int
}

func NewIntegerTimedSincResampler[T Sample](srIn, srOut, quantum, taps int) (s *IntegerTimedSincResampler[T]) {
	if quantum != RoundUpPow2(quantum) {
		panic("quantum must be a power of 2")
	}

	s = new(IntegerTimedSincResampler[T])
	ratio := Ffdiv(srIn, srOut)
	s.ratio = ratio
	s.invRatio = 1 / ratio
	s.outStep = fixed64(s.invRatio * fixedPointOne)
	// downsampling accumulates ratio input samples per output sample
	// rescale output to maintain unity gain
	s.scaleFactor = min(s.invRatio, 1)
	// when upsampling, scale sinc inputs to low-pass at lower input rate
	// avoids aliasing
	s.sincFactor = min(s.ratio, 1)

	// TODO does taps need to scale up when sinc is widened due to resample ratio?
	//taps = Fmul(taps, s.ratio)

	s.out = make([]T, RoundUpPow2(max(taps, quantum)*4))
	s.delay = CeiledDivide(taps, 2)
	s.halfTaps = CeiledDivide(fixed64(taps<<fixedPointShift), 2)
	s.invFilterWidth = 1 / float64(taps) / s.sincFactor
	s.quantum = quantum
	s.logQuantum = tzcnt(quantum)
	return s
}

func (s *IntegerTimedSincResampler[T]) Process(in []T) {
	for _, input := range in {
		_ = input // TODO remove
		// convert fixed point output sample time to floating point approximation
		outPos := float64(s.outIdx) / float64(fixedPointOne)

		// TODO is filter length in input samples or output samples?
		// accumulate contribution of this input sample to a patch the size of the filter
		// and deposit output samples at integer slice indices
		outMin, outMax := int((s.outIdx-s.halfTaps)>>fixedPointShift), int((s.outIdx+s.halfTaps)>>fixedPointShift)

		// TODO wrap output index to avoid float precision issues at very high counts
		s.outIdx += s.outStep

		// center a windowed sinc on each output sample and calculate the weighted
		// contribution of the current input sample against that sinc window
		for oi := outMin; oi <= outMax; oi++ {
			filt := T(windowedSinc(
				(outPos-float64(oi))*s.sincFactor,
				float64(s.delay)*s.sincFactor,
				s.invFilterWidth),
			)
			// wrap into output buffer
			// TODO consider proper delay offset
			s.out[(oi+s.delay)&(len(s.out)-1)] += filt * input * T(s.scaleFactor)
		}
	}
}

func (s *IntegerTimedSincResampler[T]) Read(into []T) int {
	n := len(into)
	for nextOutChunkIdx := int(s.outIdx >> s.logQuantum >> fixedPointShift); s.quantumIdx < nextOutChunkIdx && len(into) >= s.quantum; s.quantumIdx++ {
		wrapped := (s.quantumIdx << s.logQuantum) & (len(s.out) - 1)
		chunk := s.out[wrapped:][:s.quantum]
		into = into[copy(into, chunk):]
		_memClr(chunk)
	}
	return n - len(into)
}

type OnlineSincResampler[T Sample] struct {
	out    []T
	inIdx  int
	outIdx float64
	ratio, invRatio,
	scaleFactor, sincFactor float64
	halfTaps       int
	invFilterWidth float64

	quantum    int
	invQuantum float64
}

func NewOnlineSincResampler[T Sample](quantum int, ratio float64, taps int) (s *OnlineSincResampler[T]) {
	if quantum != RoundUpPow2(quantum) {
		panic("quantum must be a power of 2")
	}

	s = new(OnlineSincResampler[T])
	s.ratio = ratio
	s.invRatio = 1 / ratio
	// downsampling accumulates ratio input samples per output sample
	// rescale output to maintain unity gain
	s.scaleFactor = min(s.invRatio, 1)
	// when upsampling, scale sinc inputs to low-pass at lower input rate
	// avoids aliasing
	s.sincFactor = min(s.ratio, 1)

	// TODO does taps need to scale up when sinc is widened due to resample ratio?
	//taps = Fmul(taps, s.ratio)

	s.out = make([]T, RoundUpPow2(max(taps, quantum)*4))
	s.halfTaps = CeiledDivide(taps, 2)
	s.invFilterWidth = 1 / float64(taps) / s.sincFactor
	s.quantum = quantum
	s.invQuantum = 1 / float64(quantum)
	return s
}

func (s *OnlineSincResampler[T]) Process(in []T) {
	// for each input sample:
	// rescale input into output float coordinates
	// convert copy to integer to get unwrapped nearest output sample (nos) index
	// for range outputSamples in nos - filtWidth/2 to nos + filtWidth/2
	// compute float distance between input and output float coordinates
	// compute windowed sinc coefficient for sinc window centered at output sample
	// scale input value by coefficient, add to output sample
	// output is delayed by filtWidth/2 samples, maybe +1
	// each output sample linearly combines ~len(in) input history samples
	// with the same number of computed sinc coefficients
	for _, input := range in {
		_ = input // TODO remove
		// scale from input sample locations to output locations
		outPos := float64(s.inIdx) * s.invRatio
		// deposit output samples at integer slice indices
		outIdx := int(outPos)
		s.inIdx++ // keep track of running input sample index

		// accumulate contribution of this input sample to a patch the size of the filter
		// TODO is filter length in input samples or output samples?
		outMin, outMax := outIdx-s.halfTaps, outIdx+s.halfTaps

		// center a windowed sinc on each output sample and calculate the weighted
		// contribution of the current input sample against that sinc window
		for oi := outMin; oi <= outMax; oi++ {
			filt := T(windowedSinc(
				(outPos-float64(oi))*s.sincFactor,
				float64(s.halfTaps)*s.sincFactor,
				s.invFilterWidth),
			)
			// wrap into output buffer
			// TODO consider proper delay offset
			s.out[(oi+s.halfTaps)&(len(s.out)-1)] += filt * input * T(s.scaleFactor)
		}
	}
	// TODO wrap input/output indices to avoid float precision issues at very high counts
}

func (s *OnlineSincResampler[T]) Read(into []T) int {
	n := len(into)
	nextOutPos := float64(s.inIdx) * s.invRatio
	nextOutChunkIdx := int(nextOutPos * s.invQuantum)

pushOutput:
	lastOutChunkIdx := int(s.outIdx * s.invQuantum)

	if lastOutChunkIdx < nextOutChunkIdx && len(into) >= s.quantum {
		wrapped := (lastOutChunkIdx * s.quantum) & (len(s.out) - 1)
		chunk := s.out[wrapped:][:s.quantum]
		into = into[copy(into, chunk):]
		_memClr(chunk) // TODO clearing inappropriately or something, still
		s.outIdx += float64(s.quantum)
		goto pushOutput
	}
	return n - len(into)
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

func (s *OnlineSincResampler[T]) putCoefs(in int) {
	oi := 0
inLoop:
	for range in {
		outPos := float64(s.inIdx) * s.invRatio
		base := float64(int(outPos) - s.halfTaps)
		outMax := oi + s.halfTaps*2

		s.inIdx++

		// center a windowed sinc on each output sample and calculate the weighted
		// contribution of the current input sample against that sinc window
		for ; oi <= outMax; oi++ {
			if oi >= len(s.out) {
				break inLoop
			}
			filt := T(windowedSinc(outPos-base, float64(s.halfTaps), s.invFilterWidth))
			// wrap into output buffer
			s.out[oi] = filt
			base++
		}
	}
}
