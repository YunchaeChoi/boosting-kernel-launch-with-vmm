#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

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

int main(int argc, char* argv[]) {
    size_t size = 
}