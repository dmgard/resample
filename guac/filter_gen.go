package main

import (
	"fmt"

	. "github.com/dmgard/guac"
)

func filterGen() {
	sinc()
}

func sinc() {
	sincF32s(8, 1)
}

func fnName(name string, ln int, unroll int) string {
	return fmt.Sprintf(name+"_%dx%d", ln, unroll)
}

func sincF32s(ln, unroll int) {
	// TODO
	var p struct {
		Out Reg[[]float32]

		Xmin, Xmax Reg[float32]
	}
	Func(fnName("SincF32s", ln, unroll), NOSPLIT, &p)

	//monoInts := R[int](ln, unroll).Load(Data("monoInts", 0, 1, 2, 3, 4, 5, 6, 7))
	//for i := 0; i < unroll; i++ {
	//	monoInts.SwizzledUnrolls(i).Add(R[int](ln).Broadcast(i * ln))
	//}
	//pi := R[float32]().Load(Data[float32]("pi", math.Pi))
	//
	//lenF32 := R[float32]()
	//out := Iter[float32](&p.Out)
	//Convert(Len[float32, int](out).Load(), lenF32)
	//
	//_scale := R[float32]().Mul(p.Xmin.Load().Sub(&p.Xmax)).Div(lenF32)
	//scale := R[float32](ln).Broadcast(_scale)

	// scale is pi * (Xmax-Xmin) / float32(len(Out))
	// base := scale * [0 1 2 3 4 5 6 7 ... unrolledElemCount] + Xmin in registers
	// index := [0 0 0 0 ... ] float32s
	// offset := Broadcast(unrolledElemCount).SwizzledUnrolls(unrollCount...)

	// for Output
	// inPoints := base + index * scale
	// divX := 1 / inPoints
	// out = sin(inPoints) * divX
	// index += unrolledElemCount

	// set Output[0] == 1.0

	ZeroUpper()
	Ret()
}
