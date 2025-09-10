package resample

import (
	"fmt"
	"math"
	"testing"
	"unsafe"
)

func BenchmarkResample(b *testing.B) {
	BenchmarkScalarResample(b)
	BenchmarkAvxResample(b)
}

func BenchmarkScalarResample(b *testing.B) {
	type T = float32
	b.Run("48_441", func(b *testing.B) {
		const (
			srIn, srOut = 48000, 44100
			quantum     = 64
		)
		s := make([]T, quantum)

		for taps := 16; taps <= 256; taps += 8 {
			tail := printResampleSuffix(srIn, srOut, quantum, taps)
			benchNode(b, "node=offlineSinc/"+tail, New[T](srIn, srOut, taps).Process, s)
			benchNode(b, "node=integerSinc/"+tail, NewIntegerTimedSincResampler[T](srIn, srOut, quantum, taps).Process, s)
			benchNode(b, "node=onlineSinc/"+tail, NewOnlineSincResampler[T](quantum, Ffdiv(srIn, srOut), taps).Process, s)
		}
	})
	b.Run("48111_47892", func(b *testing.B) {
		const (
			srIn, srOut = 48111, 47892
			quantum     = 64
		)
		s := make([]T, quantum)

		for taps := 16; taps <= 256; taps += 8 {
			tail := printResampleSuffix(srIn, srOut, quantum, taps)
			benchNode(b, "node=offlineSinc/"+tail, New[T](srIn, srOut, taps).Process, s)
			benchNode(b, "node=integerSinc/"+tail, NewIntegerTimedSincResampler[T](srIn, srOut, quantum, taps).Process, s)
			benchNode(b, "node=onlineSinc/"+tail, NewOnlineSincResampler[T](quantum, Ffdiv(srIn, srOut), taps).Process, s)
		}
	})
}

func BenchmarkAvxResample(b *testing.B) {
	type T = float32
	defer func() {
		if err := recover(); err != nil {
			b.Fatal(err)
		}
	}()
	b.Run("48_441", func(b *testing.B) {
		const (
			srIn, srOut = 48000, 44100
			quantum     = 64
		)
		s := make([]T, quantum)

		for taps := 16; taps <= 256; taps += 8 {
			tail := printResampleSuffix(srIn, srOut, quantum, taps)
			benchNode(b, "node=avx512/"+tail, NewSIMD[T](srIn, srOut, taps).Process, s)
		}
	})
	b.Run("48111_47892", func(b *testing.B) {
		const (
			srIn, srOut = 48111, 47892
			quantum     = 64
		)
		s := make([]T, quantum)

		for taps := 16; taps <= 256; taps += 8 {
			tail := printResampleSuffix(srIn, srOut, quantum, taps)
			benchNode(b, "node=avx512/"+tail, NewSIMD[T](srIn, srOut, taps).Process, s)
		}
	})
}

func printResampleSuffix(srIn int, srOut int, quantum int, taps int) string {
	return fmt.Sprintf("ratio=%d_%d/quantum=%d/taps=%d",
		srIn, srOut, quantum, taps)
}

func _BenchmarkAvxResample(b *testing.B) {
	// TODO ??
	// math.MaxUint64/(outRate*simdVecLen) * inRate + 1 (for rounding). Indicates output register advance on SubsampleIdx overflow

	coefs := make([]float32, 256*160)
	b.Run("F32", func(b *testing.B) {
		names := sliceOf("simd=AVX", "simd=AVX512")
		taps := make([][]int, 2)
		taps[0] = make([]int, 7)
		for i := range taps[0] {
			taps[0][i] = 8 * (i + 2)
		}
		taps[1] = make([]int, 15)
		for i := range taps[1] {
			taps[1][i] = 16 * (i + 2)
		}

		ffns := sliceOf(
			sliceOf(
				ResampleFixedF32_8x2,
				ResampleFixedF32_8x3,
				ResampleFixedF32_8x4,
				ResampleFixedF32_8x5,
				ResampleFixedF32_8x6,
				ResampleFixedF32_8x7,
				ResampleFixedF32_8x8,
			),
			sliceOf(
				ResampleFixedF32_16x2,
				ResampleFixedF32_16x3,
				ResampleFixedF32_16x4,
				ResampleFixedF32_16x5,
				ResampleFixedF32_16x6,
				ResampleFixedF32_16x7,
				ResampleFixedF32_16x8,
				ResampleFixedF32_16x9,
				ResampleFixedF32_16x10,
				ResampleFixedF32_16x11,
				ResampleFixedF32_16x12,
				ResampleFixedF32_16x13,
				ResampleFixedF32_16x14,
				ResampleFixedF32_16x15,
				ResampleFixedF32_16x16,
			),
		)

		for i, fns := range ffns {
			for z, fn := range fns {
				layout := fmt.Sprintf("/layout=%dx%d", (i+1)*8, z+2)
				b.Run(names[i]+"/version=2"+layout, func(b *testing.B) {
					var coefIdx, outIdx int
					benchNode(b, printResampleSuffix(48000, 44100, taps[i][z], taps[i][z]),
						func(s []float32) {
							coefIdx, outIdx = fn(s, s, coefs, coefIdx, outIdx, fixedPointOne*480/441)
						}, make([]float32, taps[i][z]))
					coefIdx, outIdx = 0, 0
					benchNode(b, printResampleSuffix(48000, 44000, taps[i][z], taps[i][z]),
						func(s []float32) {
							coefIdx, outIdx = fn(s, s, coefs, coefIdx, outIdx, fixedPointOne*48/44)
						}, make([]float32, taps[i][z]))
					coefIdx, outIdx = 0, 0
					benchNode(b, printResampleSuffix(48000, 48000, taps[i][z], taps[i][z]),
						func(s []float32) {
							coefIdx, outIdx = fn(s, s, coefs, coefIdx, outIdx, fixedPointOne)
						}, make([]float32, taps[i][z]))
				})
			}
		}

		for i, fn := range sliceOf(ResampleF32x64_8x8, ResampleF32x64_16x16) {
			taps := sliceOf(64, 256)
			layout := sliceOf("/layout=8x8", "/layout=16x16")
			b.Run(names[i]+"/version=1"+layout[i], func(b *testing.B) {
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
