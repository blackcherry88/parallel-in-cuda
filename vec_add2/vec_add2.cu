#include <wb.h>
#include <cuda_runtime.h>


#define BLOCK_SIZE 256

inline int ceil(int a, int b){
    return int((a + b - 1) / b);
}

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len){
        out[idx] = in1[idx] + in2[idx];
    }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    printf("The input length is %d\n", inputLength);

	wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void**) &deviceInput1, sizeof(float) * inputLength);
    cudaMalloc((void**) &deviceInput2, sizeof(float) * inputLength);
    cudaMalloc((void**) &deviceOutput, sizeof(float) * inputLength);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput1, hostInput1, sizeof(float) * inputLength, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, sizeof(float) * inputLength, cudaMemcpyHostToDevice);


    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    int gridDim = ceil(inputLength, BLOCK_SIZE);
    int blockDim = BLOCK_SIZE;
    printf("Grid %d dim %d\n", gridDim, blockDim);
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    vecAdd<<<gridDim, blockDim>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * inputLength, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    wbTime_stop(GPU, "Freeing GPU Memory");

    // for(int index = 0; index < inputLength; index ++){
    //    printf("%f\t%f\t%f\n", hostInput1[index],hostInput2[index],hostOutput[index]);
    // }

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

