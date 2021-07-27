#include<stdio.h>
#include<cuda.h>

#define N 10000

__global__ void add(int* a, int* b, int* c)
{
	unsigned int y = blockDim.x * blockIdx.x + threadIdx.x;
	if (y < N)
		c[y] = a[y] + b[y];
}

int check(int* a, int* b, int* c)
{
	for (int i = 0; i < N; i++)
	{
		if (c[i] != a[i] + b[i])
			return 0;
	}
	return 1;
}

int main()
{
	int* h_a, * h_b, * h_c;
	int* d_a, * d_b, * d_c;

	// allocating memory on host	
	h_a = (int*)malloc(N * sizeof(int));
	h_b = (int*)malloc(N * sizeof(int));
	h_c = (int*)malloc(N * sizeof(int));

	//assigning random values to the array elements
	for (int i = 0; i < N; i++)
	{
		h_a[i] = 1;
		h_b[i] = 2;
	}


	//assigning memory on the device	
	cudaMalloc((void**)&d_a, N * sizeof(int));
	cudaMalloc((void**)&d_b, N * sizeof(int));
	cudaMalloc((void**)&d_c, N * sizeof(int));

	//copying elements from host to device
	cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);


	//calling the function and calculating the sum on device
	add << < N / 1024 + 1, 1024 >> > (d_a, d_b, d_c);

	//copying the result to host memory
	cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	if (check(h_a, h_b, h_c))
		printf("Array sum is correct\n");
	else
		printf("Array sum is incorrect\n");

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(h_a);
	free(h_b);
	free(h_c);

}