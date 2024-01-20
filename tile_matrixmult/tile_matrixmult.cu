#include <wb.h>
#include <cuda_runtime.h>


#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)


inline int ceil(int a, int b){
    return int((a + b - 1) / b);
}


// Compute C = A * B
const int TILE_WIDTH = 16;

__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    __shared__ float tA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tB[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    float p = 0;
    for (int k = 0; k < numAColumns; k+=TILE_WIDTH) {
        // A[y_block][k_block]
        if (y >= numARows || (k+tx) >=numAColumns) {
            tA[ty][tx] = 0.0;
        } else {
            tA[ty][tx] = A[y * numAColumns + k + tx];
        }
        // B[k_block][x_block]
        if ((k+ty)>= numBRows || x >= numBColumns) {
            tB[ty][tx] = 0.0;
        } else {
            tB[ty][tx] = B[(k+ty)*numBColumns + x];
        }
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) {
            p += tA[ty][i] * tB[i][tx];
        }
        __syncthreads();
    }

    if (y >= numCRows || x >= numCColumns) {
        return;
    }
    C[y*numCColumns + x] = p;
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    size_t c_size = sizeof(float) * numCRows * numCColumns;
    hostC = (float *)malloc(c_size);

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    size_t a_size = sizeof(float) * numARows * numAColumns;
    size_t b_size = sizeof(float) * numBRows * numBColumns;

    wbCheck(cudaMalloc((void **)&deviceA, a_size));
    wbCheck(cudaMalloc((void **)&deviceB, b_size));
    wbCheck(cudaMalloc((void **)&deviceC, c_size));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    wbCheck(cudaMemcpy(deviceA, hostA, a_size, cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceB, hostB, b_size, cudaMemcpyHostToDevice));
    

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    const int block_size = TILE_WIDTH;
    dim3 gridDim(ceil(numCColumns, block_size), ceil(numCRows, block_size));
    dim3 blockDim(block_size, block_size);


    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, 
        numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

   cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    wbCheck(cudaMemcpy(hostC, deviceC, c_size, cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    wbCheck(cudaFree(deviceA));
    wbCheck(cudaFree(deviceB));
    wbCheck(cudaFree(deviceC));

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

