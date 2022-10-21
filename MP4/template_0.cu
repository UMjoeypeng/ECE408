#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define KERNEL_WIDTH 3
__constant__ float kernel[KERNEL_WIDTH][KERNEL_WIDTH][KERNEL_WIDTH];
//@@ Define constant memory for device kernel here

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int idx = z * x_size * y_size + y * x_size + x;
  float Pvalue = 0.0f;
  if(x >= x_size || y >= y_size || z >= z_size){
    return;
  }
  for(int i = -1; i < 2; i++){
    for(int j = -1; j < 2; j++){
      for(int k = -1; k < 2; k ++){
        if((x + i < 0 || x + i >= x_size) 
        || (y + j < 0 || y + j >= y_size)
        || (z + k < 0 || z + k >= z_size)){
          continue;
        }
        Pvalue += input[(z + k) * x_size * y_size + (y + j) * x_size + x + i] * kernel[k + 1][j + 1][i + 1];
      }
    }
  }
  output[idx] = Pvalue;
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int sizeInput = x_size * y_size * z_size * sizeof(float);
  cudaMalloc((void **)&deviceInput, sizeInput);
  cudaMalloc((void **)&deviceOutput, sizeInput);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, sizeInput, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(kernel, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU"); 

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  int block_width = 2;
  dim3 dimGrid(ceil((1.0 * x_size) / block_width), ceil((1.0 * y_size) / block_width), ceil((1.0 * z_size) / block_width));
  dim3 dimBlock(block_width, block_width, block_width);
  
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, sizeInput, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);
  // if(inputLength < 600){
  //   for(int i = 0; i < inputLength; i++){
  //     printf("%f ", hostOutput[i]);
  //   }
  // }

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
