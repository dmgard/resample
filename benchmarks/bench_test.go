package resample

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/dmgard/resample"
	"github.com/zeozeozeo/gomplerate"
)

func BenchmarkResampleAll(b *testing.B) {
	b.Run("simd=scalar/pkg=resample", BenchmarkScalarResample)
	b.Run("simd=best/pkg=resample", BenchmarkAvxResample)
	b.Run("simd=scalar/pkg=gomplerate", BenchmarkGomplerate)
}

func BenchmarkScalarResample(b *testing.B) {
	type T = float32
	b.Run("48_441", func(b *testing.B) {
		const (
			srIn, srOut = 48000, 44100
			quantum     = 64
		)
		s := make([]T, quantum)

		for taps := 16; taps < 496; taps += 16 {
			tail := printResampleSuffix(srIn, srOut, quantum, taps)
			benchNode(b, "node=offlineSinc/"+tail, resample.New[T](srIn, srOut, taps).Process, s)
			benchNode(b, "node=integerSinc/"+tail, resample.NewIntegerTimedSincResampler[T](srIn, srOut, quantum, taps).Process, s)
		}
	})
	b.Run("48111_47892", func(b *testing.B) {
		const (
			srIn, srOut = 48111, 47892
			quantum     = 64
		)
		s := make([]T, quantum)

		for taps := 16; taps < 496; taps += 16 {
			tail := printResampleSuffix(srIn, srOut, quantum, taps)
			benchNode(b, "node=offlineSinc/"+tail, resample.New[T](srIn, srOut, taps).Process, s)
			benchNode(b, "node=integerSinc/"+tail, resample.NewIntegerTimedSincResampler[T](srIn, srOut, quantum, taps).Process, s)
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

		for taps := 16; taps < 496; taps += 16 {
			tail := printResampleSuffix(srIn, srOut, quantum, taps)
			benchNode(b, "node=avx512/"+tail, resample.NewSIMD[T](srIn, srOut, taps).Process, s)
		}
	})
	b.Run("48111_47892", func(b *testing.B) {
		const (
			srIn, srOut = 48111, 47892
			quantum     = 64
		)
		s := make([]T, quantum)

		for taps := 16; taps < 496; taps += 16 {
			tail := printResampleSuffix(srIn, srOut, quantum, taps)
			benchNode(b, "node=avx512/"+tail, resample.NewSIMD[T](srIn, srOut, taps).Process, s)
		}
	})
}

func BenchmarkGomplerate(b *testing.B) {
	type T = float64
	doGomple := func(srIn, srOut, quantum int) {
		s := make([]T, quantum)

		gr, err := gomplerate.NewResampler(1, srIn, srOut)

		if err != nil {
			b.Fatal(err)
		}

		process := func(d []T) {
			gr.ResampleFloat64(d)
		}
		tail := fmt.Sprintf("sr=%d_%d/", srIn, srOut)
		benchNode(b, "node=gomplerate/"+tail, process, s)
	}
	doGomple(48000, 44100, 1<<17)
	doGomple(48111, 47892, 1<<17)
}

//func BenchmarkZafResample(b *testing.B) {
//	type T = float32
//	do := func(srIn, srOut float64, quantum, quality int) {
//		s := make([]T, quantum)
//
//		gr, err := zafResample.New(io.Discard, srIn, srOut, 1, zafResample.F32, quality)
//
//		if err != nil {
//			b.Fatal(err)
//		}
//
//		process := func(d []T) {
//			gr.Write(resample.SliceCast[byte](d))
//		}
//		tail := fmt.Sprintf("sr=%d_%d/taps=%d/", srIn, srOut, []int{
//			zafResample.Quick:     16,
//			zafResample.LowQ:      32,
//			zafResample.MediumQ:   64,
//			zafResample.HighQ:     128,
//			zafResample.VeryHighQ: 256,
//		}[quality])
//		benchNode(b, "node=gomplerate/"+tail, process, s)
//	}
//	for quality := range []int{zafResample.Quick,
//		zafResample.LowQ,
//		zafResample.MediumQ,
//		zafResample.HighQ,
//		zafResample.VeryHighQ} {
//		do(48000, 44100, 1<<17, quality)
//		do(48111, 47892, 1<<17, quality)
//	}
//}

func printResampleSuffix(srIn int, srOut int, quantum int, taps int) string {
	return fmt.Sprintf("ratio=%d_%d/quantum=%d/taps=%d",
		srIn, srOut, quantum, taps)
}

func benchNode[T float32 | float64](b *testing.B, name string, process func([]T), samples []T) {
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
