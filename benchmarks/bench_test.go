package resample

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/dmgard/resample"
	"github.com/faiface/beep"
	"github.com/zeozeozeo/gomplerate"
)

func BenchmarkResampleAll(b *testing.B) {
	b.Run("simd=scalar/pkg=beep", BenchmarkBeep)
	b.Run("simd=best/pkg=resample", BenchmarkResample)
	b.Run("simd=scalar/pkg=gomplerate", BenchmarkGomplerate)
}

func BenchmarkResample(b *testing.B) {
	type T = float32

	doFn := func(srIn, srOut, quantum int) {
		b.Run(fmt.Sprintf("ratio=%d_%d/quantum=%d", srIn, srOut, quantum), func(b *testing.B) {
			s := make([]T, quantum)

			for taps := 16; taps <= 480; taps += 16 {
				tail := fmt.Sprintf("taps=%d", taps)
				benchNode(b, "node=avx512/"+tail, resample.New[T](srIn, srOut, taps).Process, s)
			}
		})
	}
	doFn(48000, 44100, 64)
	doFn(48111, 47892, 64)
	//doFn(48000, 44100, 512)
	//doFn(48111, 47892, 512)
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
		tail := fmt.Sprintf("ratio=%d_%d/quantum=%d/", srIn, srOut, quantum)
		benchNode(b, "node=gomplerate/"+tail, process, s)
	}
	doGomple(48000, 44100, 64)
	doGomple(48111, 47892, 64)
}

func BenchmarkBeep(b *testing.B) {
	type T = float64
	do := func(srIn, srOut, quantum, quality int) {
		s := make([]T, quantum)

		gr := beep.Resample(quality, beep.SampleRate(srIn), beep.SampleRate(srOut), beep.Silence(-1))

		samples := make([][2]float64, quantum)

		process := func(d []T) {
			gr.Stream(samples)
		}

		// TODO do real quality comparisons to establish principled relationship between
		// quality and taps. Intuition is that an N-quality lagrangian interpolation
		// performs N*N computations and increasingly approximates a windowed sinc with
		// N*N zero-crossings/taps

		tail := fmt.Sprintf("ratio=%d_%d/taps=%d/quantum=%d", srIn, srOut, quality*quality, quantum)
		benchNode(b, "node=beep/"+tail, process, s)
	}
	for quality := 1; quality <= 16; quality <<= 1 {
		do(48000, 44100, 64, quality)
		do(48111, 47892, 64, quality)
	}
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
//		tail := fmt.Sprintf("ratio=%d_%d/taps=%d/", srIn, srOut, []int{
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

func benchNode[T float32 | float64](b *testing.B, name string, process func([]T), samples []T) {
	b.Run(name, func(b *testing.B) {
		var t T
		b.SetBytes(int64(len(samples)) * int64(unsafe.Sizeof(t)))

		for n := 0; n < b.N; n++ {
			process(samples)
		}

		lnScale := 1
		switch any(*new(T)).(type) {
		case [2]float64:
			lnScale = 2
		}

		scaledLn := len(samples) * lnScale

		nsPerSample := float64(b.Elapsed().Nanoseconds()) / float64(b.N) / float64(scaledLn)
		b.ReportMetric(nsPerSample, "ns/sample")
		b.ReportMetric(1000000000./48000./nsPerSample, "xRt_48k")
	})
}
