package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"

	"github.com/dmgard/resample"
)

func main() {
	const inSr = 96000
	const outSr = 44100
	const ln = 8 * inSr / 256 * 256
	lss := resample.LogSweptSine[float32](ln, 0., ln/2)
	//lss := resample.LogSweptSine2[float32](inSr, 8*time.Second)

	filename := "groundTruth96k"
	os.WriteFile(filename+".f4", resample.SliceCast[byte](lss), 0666)

	cmd := exec.Command("sox", "-r", "96000", "-c", "1", filename+".f4", "-n", "spectrogram", "-o", "\""+filename+".png\"")
	log.Println(cmd.String())

	if res, err := cmd.Output(); err != nil {
		log.Println(string(res))
		log.Println(err)
	}
	outRarg := strconv.Itoa(outSr)

	out := resample.DupSized(lss)

	for taps := 16; taps <= 256; taps <<= 1 {
		r := resample.New[float32](inSr, outSr, taps)
		buf := out

		for i := 0; i < len(lss); i += taps {
			r.Process(lss[i:][:taps])
			buf = buf[r.Read(buf):]
		}

		out := out[taps:][:len(out)-len(buf)]

		filename := fmt.Sprintf("96to441x%d", taps)
		os.WriteFile(filename+".f4", resample.SliceCast[byte](out), 0666)

		cmd := exec.Command("sox", "-r", outRarg, "-c", "1", filename+".f4", "-n", "spectrogram", "-o", "\""+filename+".png\"")
		log.Println(cmd.String())
		if res, err := cmd.Output(); err != nil {
			log.Println(string(res))
			log.Println(err)
		}
	}
}

func main_48_441() {
	const inSr = 48000
	const outSr = 44100
	const ln = inSr / 256 * 256
	lss := resample.LogSweptSine[float32](ln, 0., ln/4)

	filename := "groundTruth48k"
	os.WriteFile(filename+".f4", resample.SliceCast[byte](lss), 0666)

	cmd := exec.Command("sox", "-r", "48000", "-c", "1", filename+".f4", "-n", "spectrogram", "-o", "\""+filename+".png\"")
	log.Println(cmd.String())

	if res, err := cmd.Output(); err != nil {
		log.Println(string(res))
		log.Println(err)
	}

	out := resample.DupSized(lss)

	for taps := 16; taps <= 256; taps <<= 1 {
		r := resample.New[float32](48000, 44100, taps)
		d := resample.New[float32](44100, 48000, taps)
		_, _ = r, d

		scratch := make([]float32, taps)
		buf := out

		for i := 0; i < len(lss); i += taps {
			r.Process(lss[i:][:taps])
			d.Process(scratch[:r.Read(scratch)])
			buf = buf[d.Read(buf):]
		}

		filename := fmt.Sprintf("48to441x%d", taps)
		os.WriteFile(filename+".f4", resample.SliceCast[byte](out[taps:]), 0666)

		cmd := exec.Command("sox", "-r", " 48000", "-c", "1", filename+".f4", "-n", "spectrogram", "-o", "\""+filename+".png\"")
		log.Println(cmd.String())
		if res, err := cmd.Output(); err != nil {
			log.Println(string(res))
			log.Println(err)
		}
	}
}
