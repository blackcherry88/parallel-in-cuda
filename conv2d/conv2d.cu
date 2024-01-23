#include    <iostream>
#include    <cuda_runtime.h>
#include    <wb.h>
#include    <helper_cuda.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

#define TILE_W 16
#define BLOCK_SIZE TILE_W + Mask_width - 1
__constant__ float deviceMask[Mask_width][Mask_width];


inline int ceil(int a, int b){
    return (a + b - 1) / b;
}

//@@ INSERT CODE HERE
template<int TILE_SIZE>
__global__ void conv2d(float *i_img, float*o_img, int img_x, int img_y, int img_z) 
/***
 * @param img_z is channel
*/
{
    __shared__ float T[TILE_SIZE+Mask_width-1][TILE_SIZE+Mask_width-1];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int out_y = blockIdx.y * TILE_SIZE + ty;
    int y = out_y - Mask_radius;
    int stride_y = img_x * img_z;

    int out_x = blockIdx.x * TILE_SIZE + tx;
    int x = out_x - Mask_radius;
    int stride_x = img_z;
    int channel = blockIdx.z;

    // if (tx == 0 && ty == 0) {
    //     printf("AAA (%d, %d, %d) --> (%d, %d) Block(%d) vs Tile (%d)\n", x, y, channel, out_x, out_y, blockDim.x, TILE_SIZE);
    // }

    if (x >= 0 && x < img_x &&
        y >= 0 && y < img_y) {
        T[ty][tx] = i_img[y*stride_y + x*stride_x + channel];

    } else {
        T[ty][tx] = 0.0f;
    } 
    __syncthreads();

    float output = 0.0f;
    if(tx < TILE_SIZE && ty < TILE_SIZE){
        for(int j = 0; j < Mask_width; j++){
            for(int i = 0; i < Mask_width; i++){
                output += deviceMask[j][i] * T[j+ty][i+tx];
            }
        }
        // set output
        if(out_y < img_y && out_x < img_x){
            o_img[out_y * stride_y + out_x * stride_x + channel] = output;
        }
    }
}



int main(int argc, char* argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbPPM_import(inputImageFile);

    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);


    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);


    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    // use constant memory for deviceMask
    cudaMemcpyToSymbol(deviceMask,
                       hostMaskData,
                       Mask_width * Mask_width * sizeof(float),
                       0,
                       cudaMemcpyHostToDevice
                      );
     
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    dim3 DimGrid(ceil(imageWidth, TILE_W), ceil(imageHeight, TILE_W), imageChannels);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    
    std::cout <<"begin to do GPU computation"<<std::endl;
    //@@ INSERT CODE HERE
    conv2d<TILE_W><<<DimGrid, DimBlock>>>(deviceInputImageData, deviceOutputImageData, 
                imageWidth, imageHeight, imageChannels);
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
        
    std::cout<<"check mask "<<std::endl;
    for(int row = 0; row < 5; row ++){
        for(int col = 0; col < 5; col ++){
            std::cout<<hostMaskData[row * Mask_width + col]<<", ";
        }
        std::cout<<endl;
    }

    std::cout<<"check output "<<std::endl;
    for(int row = 0; row < 5; row ++){
        for(int col = 0; col < 5; col ++){
            std::cout<<hostOutputImageData[(row * imageWidth + col) * imageChannels + 0]<<", ";
        }
        std::cout<<endl;
    }
    
    wbSolution(arg, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
