#include <stdio.h>


const int N = 1024;

__global__ void device_add(int *a, int *b, int *c, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        c[idx] = a[idx] + b[idx];
    }
}


void fill_array(int *a, int size) 
{
    for (int i = 0; i < size; i++) 
    {
        a[i] = i;
    }
}

void print_output(int *a, int *b, int *c)
{
	for (int idx = 0; idx < N; idx++)
		printf("\n %d + %d  = %d", a[idx], b[idx], c[idx]);
}

int main() 
{
    int * a, * b, *c;
    int *d_a, *d_b, *d_c;

	int size = N * sizeof(int);
    a = (int *)malloc(size);
    fill_array(a, N);
    b = (int *)malloc(size);
    fill_array(b, N);
    c = (int *)malloc(size);

    // Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int n_threads = 8;
    int n_blocks = N / n_threads;

    device_add<<<n_blocks, n_threads>>>(d_a, d_b, d_c, N);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    print_output(a, b, c);
    printf("\nStart to free memory\n");
    free(a);
    free(b);
    free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);    
}