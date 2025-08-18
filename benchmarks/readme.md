# Benchmarks

```bash
# Resample
cd 9950x3d # or whatever CPU you're using
go test ../../ -run=~ -bench=Resample -benchtime=10ms -count=6 >> resample_v1.txt

benchstat -table "/ratio" -col "/node /quantum /simd /layout" -row "/taps" -filter ".unit:(xRt_48k OR ns/sample OR MB/s OR B/op)" resample_v1.txt > benchstat_resample_v1.txt
```