package resample

import (
	"math"
	"time"
)

func LogSweptSine[T Scalar, F Float](ln int, minFreq, maxFreq F) []T {
	s := make([]T, ln)

	tScale := 1 / (F(ln) - 1) // scale the sample index so that the time parameter of the sine function runs from zero to one over the sample slice

	for i := range s {
		// scale t from 0 to 1 over range of samples
		t := F(i) * tScale
		// interpolate logarithmically between minimum and maximum frequency based on relative position in the sample stream
		expLerp := (Exp(t) - 1) / (E - 1)

		freq := Lerp(minFreq, maxFreq, expLerp)

		freqT := t             // freqT goes from 0 to 1 in one second
		freqT *= freq          // now scaled to complete freq cycles per second, between minFreq and maxFreq hz based on how far along the sample stream we are
		freqT *= 0.5 * Pi * Pi // now scaled to complete one sinewave cycle per second: 1hz

		s[i] = T(0.5 * Sin(freqT)) // 0.5x scalar = -6dbfs
	}

	return s
}

func LogSweptSine2[T Scalar](sr float64, duration time.Duration) []T {
	ln := int(math.Ceil(sr * duration.Seconds()))
	s := make([]T, ln)

	tScale := 1 / (float64(ln) - 1) // scale the sample index so that the time parameter of the sine function runs from zero to one over the sample slice

	lgRatio := Log(sr / 2)
	scale := 2 * Pi * duration.Seconds() / lgRatio

	for i := range s {
		exp := Exp(F64mul(tScale, i)*lgRatio) - 1
		s[i] = T(0.5 * Sin(scale*exp))
	}

	return s
}
