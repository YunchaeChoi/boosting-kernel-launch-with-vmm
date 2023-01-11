#!/bin/bash

for i in {1..10};
do 
	nsys nvprof --print-gpu-trace ./fixed_low_level_3mm.exe 1>log.tmp && cat log.tmp | grep mm*_kernel > kernel.log 
done;
