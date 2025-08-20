package resample

import (
	"math"
	"math/bits"
	"unsafe"
)

func Resize[T any, ST ~[]T, I Integer](s *ST, lc ...I) *ST {
	lnCap := [2]I{}
	copy(lnCap[:], lc)
	switch len(lc) {
	case 0:
	case 1:
		lnCap[1] = lnCap[0]
	case 2:
	default:
		panic("too many length/capacity parameters")
	}

	if lnCap[0] > lnCap[1] {
		panic("length larger than capacity")
	}

	if cap(*s) < int(lnCap[1]) {
		s2 := make([]T, lnCap[1])
		copy(s2, *s)
		*s = s2
	}

	*s = (*s)[:lnCap[0]]
	return s
}

type (
	Scalar interface {
		Integer | Float
	}

	Integer interface {
		~int8 | ~int16 | ~int32 | ~int64 | ~int | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uint | ~uintptr
	}
	Unsigned interface {
		~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uint | ~uintptr
	}
	Float interface {
		float32 | float64
	}
	Complex interface {
		~complex64 | ~complex128
	}
)

type Errors []error

func (e Errors) Error() (s string) {
	for i := range e {
		s += e[i].Error() + "\n"
	}

	return s
}

func (e *Errors) AppendIf(err error) bool {
	if err == nil {
		return false
	}

	*e = append(*e, err)
	return true
}

func (e Errors) Return() error {
	if len(e) > 0 {
		return e
	}

	return nil
}

func Wrapped[T Scalar](v T, min, max T) T {
	// TODO think on giving up and returning mod after too many cycles
	if min > max {
		min, max = max, min
	}
	for v >= max {
		v -= max - min
	}
	for v < min {
		v += max - min
	}

	return v
}

func RoundUpPow2[T Scalar](vv T) T {
	v := uint64(vv)
	v--
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v |= v >> 32
	v++
	return T(v)
}

func RoundUpMultPow2[T Integer](v, multiple T) T {
	if multiple == 0 || multiple&(multiple-1) != 0 {
		panic("zero multiple or non-power of two")
	}

	return (v + multiple - 1) & -multiple
}

func tzcnt[I Integer](i I) int { return bits.TrailingZeros64(uint64(i)) }

func _memClr[T any](s []T) {
	for i := range s {
		s[i] = *new(T)
	}
}

func Fmul[T Scalar, F Float](s T, f F) T {
	return T(F(s) * f)
}

func F64mul[T, F Scalar](s T, f F) T {
	return T(float64(s) * float64(f))
}

func FmulCeiled[T Scalar, F Float](s T, f F) T {
	return T(math.Ceil(float64(F(s) * f)))
}

func Fdiv[T Scalar, F Scalar](s T, f F) T {
	return T(Ffdiv(s, f))
}

func Ffdiv[T Scalar, F Scalar](s T, f F) float64 {
	return float64(s) / float64(f)
}

func Ftdiv[F Float, T, S Scalar](s T, f S) F {
	return F(s) / F(f)
}

func Copysign[T Scalar, F Scalar](to T, from F) T {
	return T(math.Copysign(float64(to), float64(from)))
}

func SliceCast[T, F any](from []F) []T {
	if len(from) == 0 {
		return nil
	}
	var t T
	var f F

	newSize := (uintptr(len(from)) * unsafe.Sizeof(f)) / unsafe.Sizeof(t)

	return unsafe.Slice((*T)(unsafe.Pointer(&from[0])), newSize)
}

func CeiledDivide[T Integer](a, b T) T {
	return (a + b - 1) / b
}

func Dup[T any, ST ~[]T](s ST) ST {
	ss := make(ST, len(s))
	copy(ss, s)
	return ss
}

func ResizeMin[T any, ST ~[]T](ss ...*ST) int {
	if len(ss) == 0 {
		return 0
	}
	_min := len(*ss[0])
	for _, s := range ss {
		_min = min(_min, len(*s))
	}

	for _, s := range ss {
		*s = (*s)[:_min]
	}

	return _min
}

func Log[T Scalar](a T) T {
	return T(math.Log(float64(a)))
}

func Abs[T Scalar](a T) T {
	return T(math.Abs(float64(a)))
}

func Fma[T Scalar](a, b, c T) T {
	return T(math.FMA(float64(a), float64(b), float64(c)))
}

func Approx[T Scalar](delta T, a, b T) bool {
	return Abs(a-b) <= delta
}

const (
	Pi  = math.Pi
	E   = math.E
	Tau = 2 * Pi
)

func Cos[T Scalar](a T) T {
	return T(math.Cos(float64(a)))
}
func Tan[T Scalar](a T) T {
	return T(math.Tan(float64(a)))
}
func Atan[T Scalar](a T) T {
	return T(math.Atan(float64(a)))
}
func Sin[T Scalar](a T) T {
	return T(math.Sin(float64(a)))
}

func Exp[T Scalar](a T) T {
	return T(math.Exp(float64(a)))
}

func Pow[T Scalar](a, p T) T {
	return T(math.Pow(float64(a), float64(p)))
}
func If[T any, B ~bool](b B, t, f T) T {
	if b {
		return t
	}

	return f
}

func Sign[T Scalar](a T) T {
	return T(math.Copysign(1.0, float64(a)))
}
func Round[T Scalar](a T) T {
	return T(math.Round(float64(a)))
}

func Frac[F Float](f F) (F, F) {
	m, mf := math.Modf(float64(f))
	return F(m), F(mf)
}

func Last[T any, ST ~[]T](s ST, n int) ST {
	return s[len(s)-n:]
}

func Truncated[T any, ST ~[]T](s ST, n int) ST {
	return s[:len(s)-n]
}

func LastPtr[T any, ST ~[]T](s ST) *T {
	return &s[len(s)-1]
}

func Remove[T any, ST ~[]T](s *ST, idx int) *ST {
	*s = append((*s)[:idx], (*s)[idx+1:]...)
	return s
}

func RemoveFirstEq[T comparable, ST ~[]T](s *ST, elem T) *ST {
	for i, t := range *s {
		if elem == t {
			return Remove(s, i)
		}
	}
	return s
}

func TrimRight[T comparable, ST ~[]T](s *ST, cutset ...T) *ST {
	end := len(*s)
	MapReverse(*s, func(i int, t T) bool {
		for _, c := range cutset {
			if t == c {
				end = i
				return true
			}
		}
		return false
	})
	*s = (*s)[:end]
	return s
}

func MapReverse[T any, ST ~[]T](s ST, fs ...func(int, T) bool) ST {
	for i := len(s) - 1; i >= 0; i-- {
		for _, f := range fs {
			if !f(i, s[i]) {
				return s
			}
		}
	}

	return s
}
func DupSized[T any, ST ~[]T](s ST) ST {
	return make(ST, len(s))
}
func Clamped[T Scalar](v T, a, b T) T {
	if v > b {
		return b
	} else if v < a {
		return a
	}

	return v
}

func Clear[T any, ST ~[]T](s ST) ST {
	var t T
	for i := range s {
		s[i] = t
	}

	return s
}
func Lerp[T Scalar, F Float](start, end T, step F) T {
	return T((1-step)*F(start) + step*F(end))
}

func MapPtr[T any, ST ~[]T](s ST, fs ...func(*T)) ST {
	for i := range s {
		for _, f := range fs {
			f(&s[i])
		}
	}

	return s
}
