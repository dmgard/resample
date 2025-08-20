package resample

import (
	"math"
	"math/rand"
	"testing"
)

const plotWidth = 300
const plotHeight = 50

func TestApproximate(t *testing.T) {
	type T = float32

	const quantum = 64
	const taps = 64

	//const srIn, srOut, outOffset = 48111, 47892, quantum + taps
	//const srIn, srOut, outOffset = 48111, 44111, quantum + taps + 3
	//const srIn, srOut, outOffset = 40971, 21131, quantum + taps + taps/2 - 2
	const srIn, srOut, outOffset = 40971, 7131, quantum*4 + taps/3 + 3
	//const srIn, srOut, outOffset = 42, 7, quantum*4 + taps/2
	// TODO total delay seems to shift with resample ratio,
	// TODO probably multiple of taps, not quantum?

	rs := New[T](srIn, srOut, taps)
	us := New[T](srOut, srIn, taps)

	samples := YeqX[T](quantum + 1)[1:]
	samples = cosSignal[T](quantum, 1.)
	//samples = LogSweptSine[T](quantum, 0., 10.)
	//samples = Const[T](quantum, 1)

	const numQuanta = 100 * 2048

	output := make([]T, quantum*numQuanta)
	recovered := output
	buf := make([]T, quantum)

	for range numQuanta {
		rs.Process(samples)
		if next := buf[:rs.Read(buf)]; len(next) != 0 {
			us.Process(next)
			recovered = recovered[us.Read(recovered):]
		}
	}
	ln := len(output) - len(recovered)
	recovered = output

	trimmed := recovered[outOffset:ln]
	if idxs, deltas, avg := MaxErrorsVsRepeat(0.001, 10,
		trimmed, samples); len(idxs) > 0 {
		t.Log("Errors at: ", idxs)
		t.Logf("Errors: %3.3f", deltas)
		t.Logf("%3.3f average error. Ground truth:", avg)
		plot(t, samples)

		for i, loc := range idxs {
			t.Logf("@%d: %3.3f", loc, deltas[i])
			chunk := safeSlice(trimmed, loc/quantum*quantum, quantum)
			plot(t, chunk)
		}

		t.Fail()
	}
}

func TestFareySearch(t *testing.T) {
	for range 100 {
		r := rand.Float64()

		num, denom, numAlt, denomAlt := FareySearch(r, 1000, 10000)
		t.Logf("%4.4f : %5d/%5d -> %4.4f (%4.4g)", r, num, denom, Ffdiv(num, denom), r-Ffdiv(num, denom))
		t.Logf("%4.4f : %5d/%5d -> %4.4f (%4.4g)", r, numAlt, denomAlt, Ffdiv(numAlt, denomAlt), r-Ffdiv(numAlt, denomAlt))
	}
}

func MaxErrorsVsRepeat[T Float](delta T, count int, test, truth []T) (errorIndex []int, d []T, avg T) {
	errorIndex = make([]int, count)
	d = make([]T, count)
	total := 0
	for i, c := range test {
		t := truth[i&(len(truth)-1)]
		err := Abs(c - t)
		avg += err

		if !Approx(delta, c, t) {
			var di, smallI int
			var del, smallest T
			for di, del = range d {
				if del == 0 || del < smallest {
					smallest = del
					smallI = di
				}

				if del == 0 {
					break
				}
			}

			if err > smallest {
				d[smallI] = err
				errorIndex[smallI] = i
				total++
			}
		}
	}
	avg = Fdiv(avg, len(test))

	errorIndex = safeSlice(errorIndex, 0, total)
	d = safeSlice(d, 0, total)

	return
}
func TestPlotSincResample(t *testing.T) {
	type T = float32

	const srIn, srOut = 6, 5
	const quantum = 64
	const taps = 8

	rs := New[T](srIn, srOut, taps)
	us := New[T](srOut, srIn, taps)

	rs2 := NewIntegerTimedSincResampler[T](srIn, srOut, quantum, taps)
	us2 := NewIntegerTimedSincResampler[T](srOut, srIn, quantum, taps)
	rs3 := NewOnlineSincResampler[T](quantum, Ffdiv(srIn, srOut), taps)
	us3 := NewOnlineSincResampler[T](quantum, Ffdiv(srOut, srIn), taps)

	samples := YeqX[T](quantum + 1)[1:]
	samples = cosSignal[T](quantum, 1.)
	samples = LogSweptSine[T](quantum, 0., 10.)
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

func safeSlice[T any](s []T, at, ln int) []T {
	return s[min(max(at, 0), len(s)):min(len(s), at+ln)]
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

func TestOfflineSincResampleCoefs(t *testing.T) {
	type T = float32

	const srIn, srOut = 4, 3
	const quantum = 64
	const taps = 8

	// TODO aliasing and non-unity gain

	// filter coefficients seem to sum to oscillating values
	// maybe sum coefficients in parallel array and then scale down by coefs

	rs := New[T](srIn, srOut, taps)

	t.Log("pre-coefs:\n" + TextPlot(plotWidth, plotHeight,
		lerpPlotter[T](rs.coefs).Plot, 0, -.25,
		T(len(rs.coefs)-2), 1))

	preCoefs := Dup(rs.coefs)
	rs.putCoefs()

	t.Log("live coefs:\n" + TextPlot(plotWidth, plotHeight,
		lerpPlotter[T](rs.coefs).Plot, 0, -.25,
		T(len(rs.coefs)-2), 1))
	ApproxVec(t, 0.00001, rs.coefs, preCoefs)
}

func ApproxVec[T Float](t *testing.T, delta T, coefs, preCoefs []T) (errorIndex int) {
	for i, c := range coefs {
		if !Approx(delta, c, preCoefs[i]) {
			t.Errorf("at %d %3.3f != %3.3f", i, preCoefs[i], c)
			return i
		}
	}
	return -1
}

func TestSincFns(t *testing.T) {
	type T = float32

	const sincWidth = 5

	t.Log("Blackman-Harris window:\n" + TextPlot(plotWidth, plotHeight,
		func(t T) T {
			return blackmanHarris(t)
		}, 0, -.25,
		1, 1))
	t.Log("sinc:\n" + TextPlot(plotWidth, plotHeight,
		func(t T) T {
			return normalizedSinc(t)
		}, -sincWidth, -.25,
		sincWidth, 1))
	t.Log("windowed sinc:\n" + TextPlot(plotWidth, plotHeight,
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
	plot[T](t, s)
}

func plot[T Float](t *testing.T, s []T) {
	(*FPlotter[T])(t).Process(s)
}

type FPlotter[T Float] testing.T

func (p *FPlotter[T]) Process(samples []T) {
	var m T = 1
	m = samples[0]
	for _, s := range samples {
		m = max(m, s)
	}

	plot := TextPlot(plotWidth, plotHeight,
		lerpPlotter[T](samples).Plot, 0, -1,
		T(len(samples)-2), m)

	//p.Logf("Processing %d samples: %3.3f", len(samples), samples)
	p.Logf("%d samples", len(samples))
	p.Log("Plot:\n" + plot)
}

type lerpPlotter[T Float] []T

func (l lerpPlotter[T]) Plot(at T) T {
	idx := int(at)
	next := min(int(at)+1, len(l))
	t := at - T(idx)

	return l[idx]*(1-t) + l[next]*t
}

type PlotFn[T Scalar] func(T) T

func TextPlot[T Scalar, I Integer](x, y I, p PlotFn[T], xMin, yMin, xMax, yMax T) string {
	Range := xMax - xMin
	prec := x
	dt := Range / T(prec-1)

	// TODO multiple overlapping plots with different symbols
	// TODO this is just a special case of a general plot function that accepts inputs other than horner-calculable polynomials...

	s := make([]T, prec)

	for i := range s {
		s[i] = p(xMin + T(i)*dt)
	}
	//minY, maxY := s[0], s[0]
	//for _, ss := range s {
	//	minY = math.Min(ss, minY)
	//	maxY = math.Max(ss, maxY)
	//}

	xXtra := I(3)
	padX := x + xXtra
	// pad by one each row for newline character
	b := make([]byte, (padX)*y)
	for i := range b {
		b[i] = ' '
	}
	// set newlines
	for i := I(0); i < I(len(b)); i += padX {
		b[i] = '|'
		b[i+padX-2] = '|'
		b[i+padX-1] = '\n'
	}
	// jump by columns
	for i := I(0); i < x; i++ {
		// transform value at column into 0.0-1.0 range
		//val := 1 - (s[i]-minY)/(maxY-minY)
		val := 1 - (s[i]-yMin)/(yMax-yMin)
		if val > 1 || val < 0 {
			continue
		}
		// transform range into row index
		row := Clamped(I(val*T(y)), 0, y-1)
		// transform row index into row byte index
		row *= padX
		// compute in-row location
		loc := row + i + 1
		// X marks the spot
		b[loc] = 'X'
	}
	return string(b)
}
