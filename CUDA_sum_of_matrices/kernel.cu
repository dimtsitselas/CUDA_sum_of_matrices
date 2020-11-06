
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void addMatricesCuda(int* a, int* b, int* c, int nRows, int nCols)
{
    int baseI = (blockDim.x * blockIdx.x + threadIdx.x) * nCols;

    for (int i = baseI; i < baseI + nCols; i++)
    {
        if (i < nRows * nCols)
            c[i] = a[i] + b[i];
    }
}

void addMatrices(int* a, int* b, int* c, int nRows, int nCols)
{
    // create device pointers
    int* d_a, * d_b, * d_c; 
    
    int nBytes = nRows * nCols * sizeof(int);

    cudaMalloc(&d_a, nBytes);
    cudaMalloc(&d_b, nBytes);
    cudaMalloc(&d_c, nBytes);

    cudaMemcpy(d_a, a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, nBytes, cudaMemcpyHostToDevice);

    addMatricesCuda <<<ceil(nRows / 256.0), 256.0 >>> (d_a, d_b, d_c, nRows, nCols);

    cudaMemcpy(c, d_c, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void addMatricesHost(int* a, int* b, int* c, int nRows, int nCols)
{
    for (int i = 0; i < nRows * nCols; i++)
        c[i] = a[i] + b[i];
}

int main()
{
    int* a, * b, * c_h, * c_d, nCols, nRows;
    nCols = nRows = 1000; 

    a = (int*)malloc(nCols * nRows * sizeof(int));
    b = (int*)malloc(nCols * nRows * sizeof(int));
    c_h = (int*)malloc(nCols * nRows * sizeof(int));
    c_d = (int*)malloc(nCols * nRows * sizeof(int));

    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            if(a) a[i * nCols + j] = rand() % 100;
            if(b) b[i * nCols + j] = rand() % 100;
        }
    }

    addMatrices(a, b, c_d, nRows, nCols);
    addMatricesHost(a, b, c_h, nRows, nCols);

    for (int i = 0; i < nRows * nCols; i++)
        assert(c_d[i] == c_h[i]);

    printf("OK\n");

    return 0;
}
