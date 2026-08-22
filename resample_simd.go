//go:build goexperiment.simd

package resample

import "simd"

// TODO a more efficient kernel would
//	partition coefficients into maxRegisters chunks.
//	load one chunk into maxRegisters.
//	broadcast sample from each input channel in sequence with the same
//	cofficients loaded.
//	accumulate multiple coefficient chunks without shifting, using the same
//	input sample broadcast index for each channel.
//	- will have to balance reloading output samples and coefficients per channel with accumulating multiple output samples in registers by loading and accumulating the same chunk range from adjacent coefficient sets
//	after all chunks for all channels for the same input index sample have
//	been accumulated into output, advance coef index as normal and continue.
//

const Simd = true

func resamplerProcessSimdF32(s *Resampler[float32], in [][]float32) {
	for i := range in[0] {
		// weight contribution of this input sample to a patch the size of the filter
		// and accumulate to output samples at integer output slice indices
		// TODO might be off by one relative to floating point calculation
		// TODO need to quantize to register length or handle wrapping and edge cases
		outMin := int(s.outIdx>>fixedPointShift) - s.delay

		s.outIdx += s.outStep // + 1

		// coefs contains precomputed centered windowed sinc on each output sample
		coefs := s.coefsIdx

		for ch, out := range s.out {
			outMin := outMin // reset for each channel
			inputs := simd.BroadcastFloat32s(in[ch][i])
			coefs = s.coefsIdx

			for si := 0; si < s.taps; si += inputs.Len() {
				coef := simd.LoadFloat32s(s.coefs[coefs:])

				outCurr := out[(outMin+s.delay)&(len(out)-1):]
				inputs.MulAdd(
					coef,
					simd.LoadFloat32s(outCurr),
				).Store(outCurr)

				coefs += inputs.Len()
				outMin += inputs.Len()
			}
		}
		s.coefsIdx = coefs

		s.wrapCoefsIdx()
	}
}
