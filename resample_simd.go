//go:build goexperiment.simd

package resample

import (
	"simd/archsimd"
)

func (s *Resampler[T]) processGoSIMD(in []T) {
	switch s := any(s).(type) {
	case *Resampler[float32]:
		switch simdLevel {
		case slInvalid:
			panic("impossible")
		case slScalar:
			panic("TODO scalar fallback")
		case slSSE:
			resamplerProcess[archsimd.Float32x4](s, SliceCast[float32](in), archsimd.LoadFloat32x4Slice, archsimd.BroadcastFloat32x4)
		case slAVX:
			resamplerProcess[archsimd.Float32x8](s, SliceCast[float32](in), archsimd.LoadFloat32x8Slice, archsimd.BroadcastFloat32x8)
		case sl512:
			resamplerProcess[archsimd.Float32x16](s, SliceCast[float32](in), archsimd.LoadFloat32x16Slice, archsimd.BroadcastFloat32x16)
		}
	case float64:
		panic("TODO")
	default:
		panic("unsupported")
	}
}

type simdVec[V simdVec[V, T], T Sample] interface {
	archsimd.Float32x4 | archsimd.Float32x8 | archsimd.Float32x16 |
	archsimd.Float64x2 | archsimd.Float64x4 | archsimd.Float64x8

	StoreSlice([]T)
	MulAdd(V, V) V
	Len() int
}

type (
	st1  [1]byte
	st2  [2]byte
	st4  [4]byte
	st8  [8]byte
	st16 [16]byte
)

func resamplerProcess[V simdVec[V, T], T Sample](s *Resampler[T], in []T, load func([]T) V, bcast func(T) V) {
	vecWidth := (*new(V)).Len()

	for _, input := range in {
		// weight contribution of this input sample to a patch the size of the filter
		// and accumulate to output samples at integer output slice indices
		// TODO might be off by one relative to floating point calculation
		outMin := int(s.outIdx>>fixedPointShift) - s.delay

		s.outIdx += s.outStep * fixed64(vecWidth) // + 1

		inSample := bcast(input)

		// coefs contains precomputed centered windowed sinc on each output sample
		for range s.taps / vecWidth {
			// wrap into output buffer
			// delay output by half the filter taps so all inputs can accumulate in time

			outSlice := s.out[(outMin+s.delay)&(len(s.out)-1):]
			out := load(outSlice)

			inSample.MulAdd(load(s.coefs[s.coefsIdx:]), out).StoreSlice(outSlice)

			s.coefsIdx += vecWidth
			outMin += vecWidth
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
