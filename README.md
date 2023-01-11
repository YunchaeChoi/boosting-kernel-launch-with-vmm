# boosting-kernel-launch-with-vmm

multi_stream_fixed_low_level_3mm.cu
vs.
multi_stream_rt_3mm.cu

### 문제상황
cudaFree(d_A);
kernel<<< >> ();
위와 같은 경우, cudaFree()가 return될 떄까지 kernel이 launch되지 않는다.

### thread를 사용하면?
void *free_in_thread(void *A) {
	cudaFree((DATA_TYPE*)A);
	return NULL;
}

kernel1<<< >>>();
pthread_create(&tid[0], NULL, free_in_thread, (void*)A_gpu);
kernel2<<< >>>();
pthread_create(&tid[1], NULL, free_in_thread, (void*)B_gpu);
kernel3<<< >>>();
<img width="783" alt="multi_stream_rt" src="https://user-images.githubusercontent.com/90437552/211803657-a96f4fdf-152e-4a8c-b0e7-58bec2fa01fa.PNG">

kernel2는 곧바로 launch될 수 있다. 하지만 kernel3는 앞선 tid 0의 thread가 완료될 때까지 kernel3가 launch되지 않는다.

그렇다면?? 

## VMM
VMM을 사용하면 바로 바로 launch 시킬 수 있다.

kernel1<<< >>>();
pthread_create(&tid[0], NULL, free_in_thread, (void*)A_gpu);
kernel2<<< >>>();
pthread_create(&tid[1], NULL, free_in_thread, (void*)B_gpu);
kernel3<<< >>>();


<img width="525" alt="multi_stream_vmm" src="https://user-images.githubusercontent.com/90437552/211803641-92cf4cee-26ad-42c9-88af-7e54dbef5508.PNG">
