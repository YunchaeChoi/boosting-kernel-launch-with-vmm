# EXECUTABLE := 3mm.exe
# CUFILES := 3mm.cu
# EXECUTABLE := low_level_3mm.exe
# CUFILES := low_level_3mm.cu
EXECUTABLE := fixed_low_level_3mm.exe
CUFILES := fixed_low_level_3mm.cu
BEMPS_O := ~/bemps.o

include ../common.mk

./runtime:fixed_runtime_api_3mm.cu
	nvcc fixed_runtime_api_3mm.cu -lcuda -o runtime