#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

#define N 10000

void add(int* a, int* b, int* c)
{
        unsigned int y = blockDim[2] * blockIdx.x() + threadIdx.x();
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
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
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
        d_a = sycl::malloc_device<int>(N, q_ct1);
        d_b = sycl::malloc_device<int>(N, q_ct1);
        d_c = sycl::malloc_device<int>(N, q_ct1);

        //copying elements from host to device
        q_ct1.memcpy(d_a, h_a, N * sizeof(int)).wait();
        q_ct1.memcpy(d_b, h_b, N * sizeof(int)).wait();

        //calling the function and calculating the sum on device
	add << < N / 1024 + 1, 1024 >> > (d_a, d_b, d_c);

	//copying the result to host memory
        q_ct1.memcpy(h_c, d_c, N * sizeof(int)).wait();

        if (check(h_a, h_b, h_c))
		printf("Array sum is correct\n");
	else
		printf("Array sum is incorrect\n");

        sycl::free(d_a, q_ct1);
        sycl::free(d_b, q_ct1);
        sycl::free(d_c, q_ct1);

        free(h_a);
	free(h_b);
	free(h_c);

}