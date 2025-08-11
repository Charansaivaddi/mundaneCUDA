#include <iostream>
#include <cuda_runtime.h>

__global__
void vecAddKernel(float *h_A, float *h_B, float *h_C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n) h_C[i] = h_A[i] + h_B[i];
}


void vecAdd(float* A, float* B, float* C, int n)
{
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_C, size);

    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    float* A = new float[1000];
    float* B = new float[1000];
    float* C = new float[1000];

    int n = 1000;
    vecAdd(A, B, C, n);

}
