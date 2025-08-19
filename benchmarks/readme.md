# Benchmarks

```bash
# Resample
cd 9950x3d # or whatever CPU you're using
go test ../../ -run=~ -bench=Resample -benchtime=10ms -count=6 >> resample_v2.txt

# /quantum /layout
benchstat -table "/ratio" -col "/node /simd /version" -row "/taps" -filter ".unit:(xRt_48k OR ns/sample OR MB/s OR B/op)" resample_v2.txt > benchstat_resample_v2.txt
```