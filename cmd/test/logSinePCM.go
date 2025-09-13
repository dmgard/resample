package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"
	"time"
	"unsafe"

	"github.com/dmgard/resample"
)

func main() {
	//main_48_441[float32]()
	//main_48_441[float64]()
	main_96_441[*resample.OfflineSincResampler[float32]](resample.New, ".scalar")
	main_96_441[*resample.OfflineSincResampler[float64]](resample.New, ".scalar")
	main_96_441[*resample.SimdResampler[float32]](resample.NewSIMD, ".simd")
	//main_96_441[*resample.SimdResampler[float64]](resample.NewSIMD, ".simd")
}

type newFn[T resample.Sample, R Processor[T]] func(int, int, int) R

type Processor[T resample.Sample] interface {
	Process([]T)
	Read([]T) int
}

func main_96_441[R Processor[T], T float32 | float64](New newFn[T, R], suffix string) {
	const inSr = 96000
	const outSr = 44100
	const secs = 8
	const ln = secs * inSr / 256 * 256
	//lss := resample.LogSweptSine[T](ln, 0., ln/2)
	lss := resample.LogSweptSine2[T](inSr, secs*time.Second)

	filename := "groundTruth96k"
	pcmExt := fmt.Sprintf(suffix+".f%d", unsafe.Sizeof(*new(T)))

	os.WriteFile(filename+pcmExt, resample.SliceCast[byte](lss), 0666)

	cmd := exec.Command("sox", "-r", "96000", "-c", "1", filename+pcmExt, "-n", "spectrogram", "-o", "\""+filename+".png\"", "-z", "180")
	log.Println(cmd.String())

	if res, err := cmd.Output(); err != nil {
		log.Println(string(res))
		log.Println(err)
	}

	sweepBytes, err := os.ReadFile("infinitewave_sweep" + pcmExt)
	if err != nil {
		log.Println(err)
	} else {
		lss = resample.SliceCast[T](sweepBytes)
		if len(lss) != resample.RoundUpMultPow2(len(lss), 256) {
			lss = lss[:len(lss)/256*256]
		}
	}

	outRarg := strconv.Itoa(outSr)

	out := resample.DupSized(lss)

	for taps := 16; taps <= 512; taps <<= 1 {
		r := New(inSr, outSr, taps)
		buf := out

		for i := 0; i < len(lss); i += taps {
			r.Process(lss[i:][:taps])
			buf = buf[r.Read(buf):]
		}

		out := out[taps:][:len(out)-len(buf)]

		filename := fmt.Sprintf("96to441x%d", taps)
		os.WriteFile(filename+pcmExt, resample.SliceCast[byte](out), 0666)

		cmd := exec.Command("sox", "-r", outRarg, "-c", "1", filename+pcmExt, "-n", "spectrogram", "-o", "\""+filename+pcmExt+".png\"", "-z", "180")
		log.Println(cmd.String())
		if res, err := cmd.Output(); err != nil {
			log.Println(string(res))
			log.Println(err)
		}
	}
}

/*
sox -r 96000 -c 1 groundTruth96k.f4 -n spectrogram -o "groundTruth96k.png" -z 180
sox -r 44100 -c 1 96to441x16.f4 -n spectrogram -o "96to441x16.png" -z 180
sox -r 44100 -c 1 96to441x32.f4 -n spectrogram -o "96to441x32.png" -z 180
sox -r 44100 -c 1 96to441x64.f4 -n spectrogram -o "96to441x64.png" -z 180
sox -r 44100 -c 1 96to441x128.f4 -n spectrogram -o "96to441x128.png" -z 180
sox -r 44100 -c 1 96to441x256.f4 -n spectrogram -o "96to441x256.png" -z 180
*/

func main_48_441[T float32 | float64]() {
	const inSr = 48000
	const outSr = 44100
	const ln = inSr / 256 * 256
	lss := resample.LogSweptSine[T](ln, 0., ln/4)

	pcmExt := fmt.Sprintf(".f%d", unsafe.Sizeof(*new(T)))
	filename := "groundTruth48k" + pcmExt
	os.WriteFile(filename, resample.SliceCast[byte](lss), 0666)

	cmd := exec.Command("sox", "-r", "48000", "-c", "1", filename, "-n", "spectrogram", "-o", "\""+filename+".png\"", "-z", "180")
	log.Println(cmd.String())

	if res, err := cmd.Output(); err != nil {
		log.Println(string(res))
		log.Println(err)
	}

	out := resample.DupSized(lss)

	for taps := 16; taps <= 256; taps <<= 1 {
		r := resample.New[T](inSr, outSr, taps)
		d := resample.New[T](outSr, inSr, taps)
		_, _ = r, d

		scratch := make([]T, taps)
		buf := out

		for i := 0; i < len(lss); i += taps {
			r.Process(lss[i:][:taps])
			d.Process(scratch[:r.Read(scratch)])
			buf = buf[d.Read(buf):]
		}

		filename := fmt.Sprintf("48to441x%d"+pcmExt, taps)
		os.WriteFile(filename, resample.SliceCast[byte](out[taps:]), 0666)

		cmd := exec.Command("sox", "-r", " 48000", "-c", "1", filename, "-n", "spectrogram", "-o", "\""+filename+".png\"", "-z", "180")
		log.Println(cmd.String())
		if res, err := cmd.Output(); err != nil {
			log.Println(string(res))
			log.Println(err)
		}
	}
}
