#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <driver_types.h>
//Performs inner product of vectors on CPU
void multiply_cpu(int *a, int *b, int* c, int N)
{
	int d[N];
	for (int i = 0; i < N; ++i)
	{
		d[i] = a[i]*b[i];
	}	
	for (int i = 0; i < N; ++i)
	{
		*c += d[i];
	}

} 
//Performs inner product of vectors on device
__global__ void multiply_kernel(int *a,int *b, int *c, int N) 
{
	int id = blockIdx.x *  blockDim.x + threadIdx.x;
	int val;
  if(id < N)
	{
    val = a[id]*b[id];
		atomicAdd(c, val);
  }
	
}
//Inner Product of vectors a = (a1,...,an), b = (b1,...,bn) can be written as a1b1+...+anbn
//This computes the vector {a1b1,...,anbn} where the sum is carried out on the CPU
__global__ void multiply_kernel_add_cpu(int *a, int* b, int* c, int N)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < N)
	{
		c[id] = a[id] * b[id];
	}	
}
//Inner Product of vectors a = (a1,...,an), b = (b1,...,bn) can be written as a1b1+...+anbn
//This computes the sum, where a is assumed to contain {a1b1,...,anbn} where the product has already
//been computed on the CPU
__global__ void multiply_cpu_add_kernel(int* a, int* b, int N)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < N)
	{
		int val = a[id];
		atomicAdd(b, val);	
	}
}

int main(int argc, char *argv[])  {
	int N = 4096;
	int T = 10, B = 1;            // threads per block and blocks per grid
	int c[1];
	int *dev_a, *dev_b, *dev_c, *dev_d;
	cudaDeviceProp device;
	cudaGetDeviceProperties(&device, 0);
	int maxthreadsperblock = device.maxThreadsPerBlock;
	long memory = device.totalGlobalMem;
	size_t shared_mem = device.sharedMemPerBlock;
	int major = device.major;
	int minor = device.minor;
	printf("Max Threads per Block: %d\n", maxthreadsperblock);
	printf("Total Global Memory: %d\n", (int)memory);
	printf("Shared Memory: %d\n", (int)shared_mem);
	printf("Major Compute Capability: %d\n", major);
	printf("Minor Compute Capability: %d\n", minor);
	do {
		printf("Enter size of array: ");
		scanf("%d", &N);
		long memory_required = 3*(long)N + 1;
		if (memory_required > memory)
		{
			printf("Too big to fit on Device!\n");
			continue;
		}
		printf("Enter number of threads per block: ");
		scanf("%d",&T);
		if (T > maxthreadsperblock)
		{
			printf("Max threads per block exceeded\n");
			continue;
		}
		printf("\nEnter number of blocks per grid: ");
		scanf("%d",&B);
		if (T * B < N) printf("Error T x B < N, try again\n");
	} while (T * B < N);

	int a[N],b[N], d[N];
	cudaEvent_t start, stop;     // using cuda events to measure time
	float elapsed_time_ms;       // which is applicable for asynchronous code also
//allocate memory on device and check for errors
	if (cudaMalloc((void**)&dev_a,N * sizeof(int)) != cudaSuccess) {	
		puts("Not enough Space on Device!");
		return 0;
	}
	if (cudaMalloc((void**)&dev_b,N * sizeof(int)) != cudaSuccess) {
		puts("Not enough Space on Device!");
		goto cleanup_a;
	}
	if (cudaMalloc((void**)&dev_c,sizeof(int)) != cudaSuccess) {
		puts("Not enough Space on Device!");
		goto cleanup_b;
	}
	if (cudaMalloc((void**)&dev_d, N * sizeof(int)) != cudaSuccess) {
		puts("Not enough Space on Device!");
		goto cleanup_c;
	}

	for(int i=0;i<N;i++) {    // load arrays with some numbers
		a[i] = i;
		b[i] = i*1;
	}
	*c = 0;
	cudaMemcpy(dev_a, a , N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b , N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c , sizeof(int),cudaMemcpyHostToDevice);
	

	cudaEventCreate( &start );     // instrument code to measure start time
	cudaEventCreate( &stop );
// Do Inner product on the Device
	cudaEventRecord( start, 0 );

	multiply_kernel<<<B,T>>>(dev_a,dev_b,dev_c, N);

	cudaMemcpy(c,dev_c,sizeof(int),cudaMemcpyDeviceToHost);

	cudaEventRecord( stop, 0 );     // instrument code to measue end time
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );

	printf("Device Inner Product: %d ", *c);
	
	printf("Time: %f ms.\n", elapsed_time_ms);  // print out execution time
	*c = 0;
//Do Inner product on the CPU
	cudaEventRecord( start, 0 );
	multiply_cpu(a,b,c, N);		
	cudaEventRecord( stop, 0 );     // instrument code to measue end time
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );

	printf("CPU Inner Product: %d ", *c);
	
	printf("Time: %f ms.\n", elapsed_time_ms);  // print out execution time
	*c = 0;
//Do Multiplication on the Device, and addition on the CPU
	cudaEventRecord( start, 0 );

	multiply_kernel_add_cpu<<<B,T>>>(dev_a,dev_b,dev_d, N);
	cudaMemcpy(d, dev_d, sizeof(int)*N, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; ++i)
	{
		*c += d[i];
	}
	cudaEventRecord( stop, 0 );     // instrument code to measue end time
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );

	printf("Multiplication on Device. Addition on CPU. Inner product: %d ", *c);
	
	printf("Time: %f ms.\n", elapsed_time_ms);  // print out execution time
	*c = 0;
//Do Multiplication on the CPU and addition on the Device
	cudaEventRecord( start, 0 );
	for (int i = 0; i < N; ++i)
	{
		d[i] = a[i] * b[i];
	}
	cudaMemcpy(dev_d, d, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, sizeof(int), cudaMemcpyHostToDevice);
	multiply_cpu_add_kernel<<<B,T>>>(dev_d, dev_c, N);
	
	cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);	
	cudaEventRecord( stop, 0 );     // instrument code to measue end time
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );

	printf("Multiplication on CPU. Addition on Device. Inner Product: %d ", *c);
	
	printf("Time: %f ms.\n", elapsed_time_ms);  // print out execution time
	// clean up

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_d);
cleanup_c:
	cudaFree(dev_c);
cleanup_b:
	cudaFree(dev_b);
cleanup_a:
	cudaFree(dev_a);	

	return 0;
}

