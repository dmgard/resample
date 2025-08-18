package resample

import (
	"math"
	"testing"

	"github.com/dmgard/oma/std/math/poly"
)

const plotWidth = 300
const plotHeight = 50

func TestPlotSincResample(t *testing.T) {
	type T = float32

	const srIn, srOut = 6, 5
	const quantum = 64
	const taps = 8

	rs := NewOfflineSincResampler[T](srIn, srOut, quantum, taps)
	us := NewOfflineSincResampler[T](srOut, srIn, quantum, taps)

	rs2 := NewIntegerTimedSincResampler[T](srIn, srOut, quantum, taps)
	us2 := NewIntegerTimedSincResampler[T](srOut, srIn, quantum, taps)
	rs3 := NewOnlineSincResampler[T](quantum, Ffdiv(srIn, srOut), taps)
	us3 := NewOnlineSincResampler[T](quantum, Ffdiv(srOut, srIn), taps)

	samples := YeqX[T](quantum + 1)[1:]
	samples = cosSignal[T](quantum, 1.)
	samples = LogSweptSine[T](quantum, 1., 10.)
	//samples = Const[T](quantum, 1)

	for range 3 {
		rs.Process(samples)
		plotRead[T](t, rs, quantum)
		us.Process(samples)
		plotRead[T](t, us, quantum)

		rs2.Process(samples)
		plotRead[T](t, rs2, quantum)
		us2.Process(samples)
		plotRead[T](t, us2, quantum)

		rs3.Process(samples)
		plotRead[T](t, rs3, quantum)
		us3.Process(samples)
		plotRead[T](t, us3, quantum)
	}
}

func YeqX[T Sample](ln int) []T {
	s := make([]T, ln)
	for i := range s {
		s[i] = T(i)
	}
	return s
}

func Const[T Sample](ln int, c T) []T {
	s := make([]T, ln)
	for i := range s {
		s[i] = c
	}
	return s
}

func cosSignal[T Sample, F Float](ln int, rate F) []T {
	s := make([]T, ln)
	invLn := 2 * rate * math.Pi / F(ln-1)
	for i := range s {
		s[i] = T(math.Cos(float64(i) * float64(invLn)))
	}
	return s
}

func LogSweptSine[T Scalar, F Float](ln int, minFreq, maxFreq F) []T {
	s := make([]T, ln)

	tScale := 1 / (F(ln) - 1) // scale the sample index so that the time parameter of the sine function runs from zero to one over the sample slice

	for i := range s {
		// scale t from 0 to 1 over range of samples
		t := F(i) * tScale
		// interpolate logarithmically between minimum and maximum frequency based on relative position in the sample stream
		freq := Lerp(minFreq, maxFreq, (Exp(t)-1)/E)
		freqT := t    // freqT goes from 0 to 1 in one second
		freqT *= Pi   // now scaled to complete one sinewave cycle per second: 1hz
		freqT *= freq // now scaled to complete freq cycles per second, between minFreq and maxFreq hz based on how far along the sample stream we are
		s[i] = T(Sin(freqT))
	}

	return s
}

func TestOfflineSincResampleCoefs(t *testing.T) {
	type T = float32

	const srIn, srOut = 4, 3
	const quantum = 64
	const taps = 8

	// TODO aliasing and non-unity gain

	// filter coefficients seem to sum to oscillating values
	// maybe sum coefficients in parallel array and then scale down by coefs

	rs := NewOfflineSincResampler[T](srIn, srOut, quantum, taps)

	t.Log("pre-coefs:\n" + poly.TextPlot(plotWidth, plotHeight,
		lerpPlotter[T](rs.coefs).Plot, 0, -.25,
		T(len(rs.coefs)-2), 1))

	preCoefs := Dup(rs.coefs)
	rs.putCoefs()

	t.Log("live coefs:\n" + poly.TextPlot(plotWidth, plotHeight,
		lerpPlotter[T](rs.coefs).Plot, 0, -.25,
		T(len(rs.coefs)-2), 1))

	for i, c := range rs.coefs {
		if !Approx(0.00001, c, preCoefs[i]) {
			t.Fatalf("coefs %d %3.3f != %3.3f", i, preCoefs[i], c)
		}
	}
}

func TestSincFns(t *testing.T) {
	type T = float32

	const sincWidth = 5

	t.Log("Blackman-Harris window:\n" + poly.TextPlot(plotWidth, plotHeight,
		func(t T) T {
			return blackmanHarris(t)
		}, 0, -.25,
		1, 1))
	t.Log("sinc:\n" + poly.TextPlot(plotWidth, plotHeight,
		func(t T) T {
			return normalizedSinc(t)
		}, -sincWidth, -.25,
		sincWidth, 1))
	t.Log("windowed sinc:\n" + poly.TextPlot(plotWidth, plotHeight,
		func(t T) T {
			return windowedSinc(t, T(sincWidth), 0.5/T(sincWidth))
		}, -sincWidth, -.25,
		sincWidth, 1))
}

type reader[T any] interface{ Read(into []T) int }

func plotRead[T Float](t *testing.T, r reader[T], n int) {
	s := make([]T, n)
	if r.Read(s) == 0 {
		return
	}
	(*FPlotter[T])(t).Process(s)
}

type FPlotter[T Float] testing.T

func (p *FPlotter[T]) Process(samples []T) {
	var m T = 1
	m = samples[0]
	for _, s := range samples {
		m = max(m, s)
	}

	plot := poly.TextPlot(plotWidth, plotHeight,
		lerpPlotter[T](samples).Plot, 0, -1,
		T(len(samples)-2), m)

	//p.Logf("Processing %d samples: %3.3f", len(samples), samples)
	p.Logf("Processing %d samples", len(samples))
	p.Log("Plot:\n" + plot)
}

type lerpPlotter[T Float] []T

func (l lerpPlotter[T]) Plot(at T) T {
	idx := int(at)
	next := min(int(at)+1, len(l))
	t := at - T(idx)

	return l[idx]*(1-t) + l[next]*t
}
