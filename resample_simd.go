//go:build goexperiment.simd

package resample

import "simd"

func resamplerProcessSimdF32(s *Resampler[float32], in ...[]float32) {
	for i := range in[0] {
		// weight contribution of this input sample to a patch the size of the filter
		// and accumulate to output samples at integer output slice indices
		// TODO might be off by one relative to floating point calculation
		outMin := int(s.outIdx>>fixedPointShift) - s.delay

		s.outIdx += s.outStep // + 1

		// coefs contains precomputed centered windowed sinc on each output sample
		for ch, out := range s.out {
			outMin := outMin // reset for each channel
			inputs := simd.BroadcastFloat32s(in[ch][i])

			for si := 0; si < s.taps; si += inputs.Len() {
				coef := simd.LoadFloat32s(s.coefs[s.coefsIdx:])

				outCurr := out[(outMin+s.delay)&(len(s.out)-1):]
				inputs.MulAdd(
					coef,
					simd.LoadFloat32s(outCurr),
				).Store(outCurr)

				s.coefsIdx += inputs.Len()
				outMin += inputs.Len()
			}
		}

		s.wrapCoefsIdx()
	}
}
