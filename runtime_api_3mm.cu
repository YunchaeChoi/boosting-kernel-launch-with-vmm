#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include <bemps.hpp>
#include <iostream>
#include "../../common/polybenchUtilFuncts.h"

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
/* default setting */
# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512

#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

typedef float DATA_TYPE;

static inline void
checkDrvError(CUresult res, const char *tok, const char *file, unsigned line)
{
    if (res != CUDA_SUCCESS) {
        const char *errStr = NULL;
        (void)cuGetErrorString(res, &errStr);
        std::cerr << file << ':' << line << ' ' << tok
                  << "failed (" << (unsigned)res << "): " << errStr << std::endl;
        abort();
    }
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);

typedef struct _vmm_struct {
	CUmemGenericAllocationHandle handle;
	CUdeviceptr ptr;
	size_t padded_size;
} vmm_struct;

vmm_struct low_level_allocation (size_t size) {
	vmm_struct vmm;
	size_t granularity;
	CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	int cur_device = 0;
	cudaGetDevice(&cur_device);
    prop.location.id = cur_device;
    CHECK_DRV(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // printf("granularity: %ld\n", granularity);
	vmm.padded_size = ((size+ granularity - 1) / granularity) * granularity;
    // printf("padded size: %ld\n", padded_size);
	CHECK_DRV(cuMemCreate(&vmm.handle, vmm.padded_size, &prop, 0));
	CHECK_DRV(cuMemAddressReserve(&vmm.ptr, vmm.padded_size, 0, 0, 0));
	CHECK_DRV(cuMemMap(vmm.ptr, vmm.padded_size, 0, vmm.handle, 0));
	CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = 0;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_DRV( cuMemSetAccess(vmm.ptr, vmm.padded_size , &accessDesc, 1ULL) );

	return vmm;
}

void low_level_free (vmm_struct vmm) {
	CHECK_DRV(cuMemUnmap(vmm.ptr, vmm.padded_size));
	CHECK_DRV(cuMemRelease(vmm.handle));
	CHECK_DRV(cuMemAddressFree(vmm.ptr, vmm.padded_size));
}

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D)
{
	int i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
			A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
			// printf("A[%d]: %f\n", i*NK + j, A[i*NK + j]);
		}
	}
  
	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			B[i*NJ + j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}
  
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NM; j++)
		{
			C[i*NM + j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}
  
	for (i = 0; i < NM; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;
		}
	}
}


void compareResults(DATA_TYPE *G, DATA_TYPE *G_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < NI; i++)
	{
		for (j=0; j < NL; j++)
		{
			if (percentDiff(G[i*NL + j], G_outputFromGpu[i*NL + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;				
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


	
__global__ void mm3_kernel1(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *E)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{
		int k;
		for(k=0; k < NK; k++)
		{
			E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
			// printf("%f\n", B[k * NJ + j]);
			// printf("E[%d]: %f\n", i * NJ + j, E[i * NJ + j]);
		}
	}
}

	
__global__ void mm3_kernel2(DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *F)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NJ) && (j < NL))
	{
		int k;
		for(k=0; k < NM; k++)
		{
			F[i * NL + j] += C[i * NM + k] * D[k * NL +j];
		}
	}
}

	
__global__ void mm3_kernel3(DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NL))
	{
		int k;
		for(k=0; k < NJ; k++)
		{
			G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
			// printf("%f\n", G[i * NL + j]);
		}
	}
}


void mm3_cpu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
	int i,j,k;
	
	/* E := A*B */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			E[i*NJ + j] = 0;
			for (k = 0; k < NK; ++k)
			{
				E[i*NJ + j] += A[i*NK + k] * B[k*NJ + j];
			}
		}
	}
		
	/* F := C*D */
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NL; j++)
		{
			F[i*NL + j] = 0;
			for (k = 0; k < NM; ++k)
			{
				F[i*NL + j] += C[i*NM + k] * D[k*NL + j];
			}
		}
	}

  	/* G := E*F */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			G[i*NL + j] = 0;
			for (k = 0; k < NJ; ++k)
			{
				G[i*NL + j] += E[i*NJ + k] * F[k*NL + j];
			}
		}
	}
}

int main(int argc, char* argv[]) {
	cudaFree(0);
	double t_start, t_end;
	t_start = rtclock();
	// cudaFree(0);

	size_t size = NI * NK * sizeof(DATA_TYPE);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NJ) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));
	dim3 grid2((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NJ/ ((float)DIM_THREAD_BLOCK_Y) )));
	dim3 grid3((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;
	DATA_TYPE* D;
	DATA_TYPE* E;
	DATA_TYPE* F;
	DATA_TYPE* G;
	DATA_TYPE* G_outputFromGpu;

	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(NJ*NM*sizeof(DATA_TYPE));
	D = (DATA_TYPE*)malloc(NM*NL*sizeof(DATA_TYPE));
	E = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	F = (DATA_TYPE*)malloc(NJ*NL*sizeof(DATA_TYPE));
	G = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));
	G_outputFromGpu = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));

	init_array(A, B, C, D);

	

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;
	DATA_TYPE *E_gpu;
	DATA_TYPE *F_gpu;
	DATA_TYPE *G_gpu;

	cudaMalloc((void **)&A_gpu, size);
	cudaMalloc((void **)&B_gpu, size);
	cudaMalloc((void **)&C_gpu, size);
	cudaMalloc((void **)&D_gpu, size);
	cudaMalloc((void **)&E_gpu, size);
	cudaMalloc((void **)&F_gpu, size);
	cudaMalloc((void **)&G_gpu, size);

	cudaError_t err;
	err = cudaGetLastError();
	if (err != 0) {
		printf("%s\n", cudaGetErrorString(err));
	}
	


	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NJ * NM, cudaMemcpyHostToDevice);
	cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * NM * NL, cudaMemcpyHostToDevice);
	/*
	cudaMemcpy(E_gpu, E, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(F_gpu, F, sizeof(DATA_TYPE) * NJ * NL, cudaMemcpyHostToDevice);
	cudaMemcpy(G_gpu, G, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyHostToDevice);	
	*/

	cudaEventRecord(start);

	mm3_kernel1<<<grid1,block>>>(A_gpu, B_gpu, E_gpu);
	
	mm3_kernel2<<<grid2,block>>>(C_gpu, D_gpu, F_gpu);
	
	mm3_kernel3<<<grid3,block>>>(E_gpu, F_gpu, G_gpu);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	cudaFree(D_gpu);
	cudaFree(E_gpu);
	cudaFree(F_gpu);
	cudaFree(G_gpu);

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float milliseconds = 0.0;
	cudaEventElapsedTime(&milliseconds, start, end);
	// printf("Elapsed Time (by cudaEvent): %fms\n", milliseconds);
	printf("%f", milliseconds);

	// cudaMemcpy(G_outputFromGpu, G_gpu, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyDeviceToHost);
	
	

	// mm3_cpu(A, B, C, D, E, F, G);
	// compareResults(G, G_outputFromGpu);

	for (int i=0; i < size/sizeof(DATA_TYPE); i++) {
		// printf("G, G_outputFromGpu (%d): %f, %f\n", i, G[i], G_outputFromGpu[i]);
	}

	// cudaFree(A_gpu);
	// cudaFree(B_gpu);
	// cudaFree(C_gpu);
	// cudaFree(D_gpu);
	// cudaFree(E_gpu);
	// cudaFree(F_gpu);
	// cudaFree(G_gpu);


	// cudaEventRecord(end);
	// cudaEventSynchronize(end);

	// float milliseconds = 0.0;
	// cudaEventElapsedTime(&milliseconds, start, end);
	// printf("Elapsed Time (by cudaEvent): %fms\n", milliseconds);

	free(A);
	free(B);
	free(C);
	free(D);
	free(E);
	free(F);
	free(G);
	free(G_outputFromGpu);

	

	cudaEventDestroy(start);
	cudaEventDestroy(end);

	t_end = rtclock();

	// printf("Total Elapsed time (by gettimeofday): %.6lfs\n", t_end - t_start);

	return 0;
}