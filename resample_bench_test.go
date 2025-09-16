package resample

import (
	"fmt"
	"testing"
	"unsafe"
)

func BenchmarkResample(b *testing.B) {
	type T = float32

	doFn := func(srIn, srOut, quantum int) {
		b.Run(fmt.Sprintf("ratio=%d_%d/quantum=%d", srIn, srOut, quantum), func(b *testing.B) {
			s := make([]T, quantum)

			for taps := 16; taps <= simdMaxFiltLens[simdLevel]; taps += 16 {
				tail := fmt.Sprintf("taps=%d", taps)
				old := simdLevel
				switch simdLevel {
				case sl512:
					if taps <= simdMaxFiltLens[simdLevel] {
						benchNode(b, "node=avx512/"+tail, New[T](srIn, srOut, taps).Process, s)
					}
					simdLevel = slAVX
					fallthrough
				case slAVX:
					if taps <= simdMaxFiltLens[simdLevel] {
						benchNode(b, "node=avx/"+tail, New[T](srIn, srOut, taps).Process, s)
					}
					simdLevel = slSSE
					fallthrough
				case slSSE:
					if false && taps <= simdMaxFiltLens[simdLevel] { // TODO SSE
						benchNode(b, "node=sse/"+tail, New[T](srIn, srOut, taps).Process, s)
					}
					simdLevel = slScalar
					fallthrough
				case slScalar:
					if taps <= simdMaxFiltLens[simdLevel] {
						benchNode(b, "node=scalar/"+tail, New[T](srIn, srOut, taps).Process, s)
					}
				}
				simdLevel = old
			}
		})
	}
	doFn(48000, 44100, 64)
	doFn(48111, 47892, 64)
}

func BenchmarkNew(b *testing.B) {
	type T = float32

	doFn := func(srIn, srOut, quantum int) {
		b.Run(fmt.Sprintf("ratio=%d_%d", srIn, srOut), func(b *testing.B) {
			for taps := 16; taps <= simdMaxFiltLens[simdLevel]; taps += 16 {
				tail := fmt.Sprintf("taps=%d", taps)
				old := simdLevel
				switch simdLevel {
				case sl512:
					if taps <= simdMaxFiltLens[simdLevel] {
						b.Run("node=avx512/"+tail, func(b *testing.B) {
							for b.Loop() {
								New[T](srIn, srOut, taps)
							}
						})
					}
					simdLevel = slAVX
					fallthrough
				case slAVX:
					if taps <= simdMaxFiltLens[simdLevel] {
						b.Run("node=avx/"+tail, func(b *testing.B) {
							for b.Loop() {
								New[T](srIn, srOut, taps)
							}
						})
					}
					simdLevel = slSSE
					fallthrough
				case slSSE:
					if false && taps <= simdMaxFiltLens[simdLevel] { // TODO SSE
						b.Run("node=sse/"+tail, func(b *testing.B) {
							for b.Loop() {
								New[T](srIn, srOut, taps)
							}
						})
					}
					simdLevel = slScalar
					fallthrough
				case slScalar:
					if taps <= simdMaxFiltLens[simdLevel] {
						b.Run("node=scalar/"+tail, func(b *testing.B) {
							for b.Loop() {
								New[T](srIn, srOut, taps)
							}
						})
					}
				}
				simdLevel = old
			}
		})
	}
	doFn(48000, 44100, 64)
	doFn(48111, 47892, 64)
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
