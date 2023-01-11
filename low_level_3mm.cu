/**
 * 3mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

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

#define GPU_DEVICE 0

#define ROUND_UP(x, n) (((x + n - 1) / n) * n)

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
/* default setting */
# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512

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

// vmm_struct* low_level_allocation (size_t size) {
// 	vmm_struct* vmm;
// 	size_t granularity;
// 	CUmemAllocationProp prop = {};
//     prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
//     prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
// 	int cur_device = 0;
// 	cudaGetDevice(&cur_device);
//     prop.location.id = cur_device;
//     CHECK_DRV(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
//     // printf("granularity: %ld\n", granularity);
// 	vmm->padded_size = ((size+ granularity - 1) / granularity) * granularity;
//     // printf("padded size: %ld\n", padded_size);
// 	CHECK_DRV(cuMemCreate(&vmm->handle, vmm->padded_size, &prop, 0));
// 	CHECK_DRV(cuMemAddressReserve(&vmm->ptr, vmm->padded_size, 0, 0, 0));
// 	CHECK_DRV(cuMemMap(vmm->ptr, vmm->padded_size, 0, vmm->handle, 0));
// 	CUmemAccessDesc accessDesc = {};
//     accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
//     accessDesc.location.id = 0;
//     accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
//     CHECK_DRV( cuMemSetAccess(vmm->ptr, vmm->padded_size , &accessDesc, 1ULL) );

// 	return vmm;
// }

void low_level_free (vmm_struct vmm) {
	CHECK_DRV(cuMemUnmap(vmm.ptr, vmm.padded_size));
	CHECK_DRV(cuMemRelease(vmm.handle));
	CHECK_DRV(cuMemAddressFree(vmm.ptr, vmm.padded_size));
}

// # define NI 2048
// # define NJ 2048
// # define NK 2048
// # define NL 2048
// # define NM 2048

// # define NI 8192
// # define NJ 8192
// # define NK 8192
// # define NL 8192
// # define NM 8192

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

cudaError_t setProp(CUmemAllocationProp *prop, bool UseCompressibleMemory)
{
    CUdevice currentDevice;
    if (cuCtxGetDevice(&currentDevice) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    memset(prop, 0, sizeof(CUmemAllocationProp));
    prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop->location.id = currentDevice;

    if (UseCompressibleMemory)
        prop->allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

    return cudaSuccess;
}

cudaError_t allocateCompressible(void **adr, size_t size, bool UseCompressibleMemory)
{
    CUmemAllocationProp prop = {};
    cudaError_t err = setProp(&prop, UseCompressibleMemory);
    if (err != cudaSuccess)
        return err;

    size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;
    size = ((size - 1) / granularity + 1) * granularity;
    CUdeviceptr dptr;
    if (cuMemAddressReserve(&dptr, size, 0, 0, 0) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    CUmemGenericAllocationHandle allocationHandle;
    if (cuMemCreate(&allocationHandle, size, &prop, 0) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    // Check if cuMemCreate was able to allocate compressible memory.
    if (UseCompressibleMemory) {
        CUmemAllocationProp allocationProp = {};
        cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);
        if (allocationProp.allocFlags.compressionType != CU_MEM_ALLOCATION_COMP_GENERIC) {
            printf("Could not allocate compressible memory... so waiving execution\n");
            exit(-1);
        }
    }

    if (cuMemMap(dptr, size, 0, allocationHandle, 0) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    if (cuMemRelease(allocationHandle) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    CUmemAccessDesc accessDescriptor;
    accessDescriptor.location.id = prop.location.id;
    accessDescriptor.location.type = prop.location.type;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    if (cuMemSetAccess(dptr, size, &accessDescriptor, 1) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    *adr = (void *)dptr;
    return cudaSuccess;
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


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
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


void mm3Cuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, DATA_TYPE* F, 
		DATA_TYPE* G, DATA_TYPE* G_outputFromGpu, int tid)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;
	DATA_TYPE *E_gpu;
	DATA_TYPE *F_gpu;
	DATA_TYPE *G_gpu;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NJ) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));
	dim3 grid2((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NJ/ ((float)DIM_THREAD_BLOCK_Y) )));
	dim3 grid3((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));

	

	bemps_begin(tid, grid1.x, grid1.y, grid1.z, block.x , block.y, block.z, sizeof(DATA_TYPE) * NI * NK * 7);

	// int flag = 0;
	// while(1) {
	// 	if (flag == 1) {
	// 		break;
	// 	}
	// 	scanf("%d", &flag);
	// 	printf("flag: %d\n", flag);
	// }

	size_t granularity = 0;
	size_t size = sizeof(float) * 512 * 512;

	printf("before low level allocation\n");
	vmm_struct VMM_A = low_level_allocation(size);
	vmm_struct VMM_B = low_level_allocation(size);
	vmm_struct VMM_C = low_level_allocation(size);
	vmm_struct VMM_D = low_level_allocation(size);
	vmm_struct VMM_E = low_level_allocation(size);
	vmm_struct VMM_F = low_level_allocation(size);
	vmm_struct VMM_G = low_level_allocation(size);
	printf("after low level allocation\n");
	
	// cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	// cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	// cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NJ * NM);
	// cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * NM * NL);
	// cudaMalloc((void **)&E_gpu, sizeof(DATA_TYPE) * NI * NJ);
	// cudaMalloc((void **)&F_gpu, sizeof(DATA_TYPE) * NJ * NL);
	// cudaMalloc((void **)&G_gpu, sizeof(DATA_TYPE) * NI * NL);

	A_gpu = (DATA_TYPE*) VMM_A.ptr;
	B_gpu = (DATA_TYPE*) VMM_B.ptr;
	C_gpu = (DATA_TYPE*) VMM_C.ptr;
	D_gpu = (DATA_TYPE*) VMM_D.ptr;
	E_gpu = (DATA_TYPE*) VMM_E.ptr;
	F_gpu = (DATA_TYPE*) VMM_F.ptr;
	G_gpu = (DATA_TYPE*) VMM_G.ptr;


	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NJ * NM, cudaMemcpyHostToDevice);
	cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * NM * NL, cudaMemcpyHostToDevice);
	cudaMemcpy(E_gpu, E, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(F_gpu, F, sizeof(DATA_TYPE) * NJ * NL, cudaMemcpyHostToDevice);
	cudaMemcpy(G_gpu, G, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyHostToDevice);	
	

	t_start = rtclock();
	mm3_kernel1<<<grid1,block>>>(A_gpu, B_gpu, E_gpu);
	cudaDeviceSynchronize();

	printf("before pre free\n");
	low_level_free(VMM_A);
	low_level_free(VMM_B);
	pre_bemps_free(tid, (long)(VMM_A.padded_size + VMM_B.padded_size));

	

	mm3_kernel2<<<grid2,block>>>(C_gpu, D_gpu, F_gpu);
	// Free C_gpu and D_Gpu here.
	cudaThreadSynchronize();
	mm3_kernel3<<<grid3,block>>>(E_gpu, F_gpu, G_gpu);
	// cudaDeviceSynchronize();
	// Free E, F_gpu
	cudaThreadSynchronize();
	t_end = rtclock();
	cudaMemcpy(G_outputFromGpu, G_gpu, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyDeviceToHost);

	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);


	
	// cudaFree(A_gpu);
	// cudaFree(B_gpu);
	// cudaFree(C_gpu);
	// cudaFree(D_gpu);
	// cudaFree(E_gpu);
	// cudaFree(F_gpu);
	// cudaFree(G_gpu);

	printf("before free\n");
	// low_level_free(VMM_A);
	// low_level_free(VMM_B);
	low_level_free(VMM_C);
	low_level_free(VMM_D);
	low_level_free(VMM_E);
	low_level_free(VMM_F);
	low_level_free(VMM_G);

	printf("end\n");
	
	bemps_free(tid);
}


int main(int argc, char** argv)
{
	int tid = atoi(argv[1]);
	double t_start, t_end;

	// int flag=-1;
	// while(1) {
	// 	if (flag == 0) {
	// 		break;
	// 	}
	// 	scanf("%d", &flag);
	// 	printf("flag: %d\n", flag);
	// }

	cudaFree(0);
	cudaDeviceSetLimit(cudaLimitStackSize, 0); // to minimize CUDA stack size (CUDA Context)

	// while(1) {
	// 	if (flag == 1) {
	// 		break;
	// 	}
	// 	scanf("%d", &flag);
	// 	printf("flag: %d\n", flag);
	// }

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

	GPU_argv_init();

	mm3Cuda(A, B, C, D, E, F, G, G_outputFromGpu, tid);

	printf("end of mm3Cuda\n");

	t_start = rtclock();

	printf("mm3 cpu starts\n");
	mm3_cpu(A, B, C, D, E, F, G);
	printf("end of mm3_cpu\n");
	
	
	t_end = rtclock();

	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(G, G_outputFromGpu);

	free(A);
	free(B);
	free(C);
	free(D);
	free(E);
	free(F);
	free(G);
	free(G_outputFromGpu);

	return 0;
}

