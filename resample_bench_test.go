package resample

import (
	"fmt"
	"math"
	"testing"
	"unsafe"
)

func BenchmarkScalarResample(b *testing.B) {
	type T = float32
	b.Run("48x441", func(b *testing.B) {
		const (
			srIn, srOut = 48000, 44100
			quantum     = 64
		)
		s := make([]T, quantum)

		for taps := 16; taps <= 256; taps <<= 1 {
			tail := printResampleSuffix(srIn, srOut, quantum, taps)

			benchNode(b, "node=offlineSinc/"+tail, NewOfflineSincResampler[T](srIn, srOut, quantum, taps).Process, s)
			benchNode(b, "node=integerSinc/"+tail, NewIntegerTimedSincResampler[T](srIn, srOut, quantum, taps).Process, s)
			benchNode(b, "node=onlineSinc/"+tail, NewOnlineSincResampler[T](quantum, Ffdiv(srIn, srOut), taps).Process, s)
		}
	})
}

func printResampleSuffix(srIn int, srOut int, quantum int, taps int) string {
	return fmt.Sprintf("ratio=%d_%d/quantum=%d/taps=%d",
		srIn, srOut, quantum, taps)
}

func sliceOf[T any](s ...T) []T { return s }

func BenchmarkAvxResample(b *testing.B) {
	// TODO ??
	// math.MaxUint64/(outRate*simdVecLen) * inRate + 1 (for rounding). Indicates output register advance on SubsampleIdx overflow

	coefs := make([]float32, 256*160)
	b.Run("F32", func(b *testing.B) {
		names := sliceOf("layout=8x8/simd=AVX", "layout=16x16/simd=AVX512")
		taps := sliceOf(64, 256)

		for i, fn := range sliceOf(ResampleF32x64_8x8, ResampleF32x64_16x16) {
			b.Run(names[i], func(b *testing.B) {
				benchNode(b, printResampleSuffix(48000, 44100, taps[i], taps[i]),
					func(s []float32) {
						fn(
							s, s, coefs,
							0, 160,
							1, math.MaxUint64/(44100*8)*48000+1, taps[i])
					}, make([]float32, taps[i]))
				benchNode(b, printResampleSuffix(48000, 44000, taps[i], taps[i]),
					func(s []float32) {
						fn(
							s, s, coefs,
							0, 12,
							1, math.MaxUint64/(44000*8)*48000+1, taps[i],
						)
					}, make([]float32, taps[i]))
				benchNode(b, printResampleSuffix(48000, 48000, taps[i], taps[i]),
					func(s []float32) {
						fn(
							s, s, coefs,
							0, 8,
							1, math.MaxUint64/(48000*8)*48000+1, taps[i],
						)
					}, make([]float32, taps[i]))
			})
		}
	})
}

func benchNode[T Sample](b *testing.B, name string, process func([]T), samples []T) {
	b.Run(name, func(b *testing.B) {
		var t T
		b.SetBytes(int64(len(samples)) * int64(unsafe.Sizeof(t)))

		for n := 0; n < b.N; n++ {
			process(samples)
		}

		nsPerSample := float64(b.Elapsed().Nanoseconds()) / float64(b.N) / float64(len(samples))
		b.ReportMetric(nsPerSample, "ns/sample")
		b.ReportMetric(1000000000./48000./nsPerSample, "xRt_48k")
	})
}
