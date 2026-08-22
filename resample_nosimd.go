//go:build !goexperiment.simd

package resample

const Simd = false

func resamplerProcessSimdF32(s *Resampler[float32], in ...[]float32) {}
