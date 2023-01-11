#!/bin/bash 
make && nvcc fixed_runtime_api_3mm.cu -lcuda -o runtime -O3
