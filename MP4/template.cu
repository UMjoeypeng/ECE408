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
#define TILE_SIZE 4

__constant__ float kernel[KERNEL_WIDTH][KERNEL_WIDTH][KERNEL_WIDTH];
//@@ Define constant memory for device kernel here

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int x = blockIdx.x * TILE_SIZE + tx;
  int y = blockIdx.y * TILE_SIZE + ty;
  int z = blockIdx.z * TILE_SIZE + tz;
  // if(x==0&&y==0&&z==0){
  //   printf("BlockDim: %d ", blockDim.x);
  // }
  int x_i = x - 1;
  int y_i = y - 1;
  int z_i = z - 1;
  __shared__ float input_ds[TILE_SIZE + 2][TILE_SIZE + 2][TILE_SIZE + 2];
  if((x_i >= 0) && (x_i < x_size) 
    && (y_i >= 0) && (y_i < y_size) 
    && (z_i >= 0) && (z_i < z_size)){
      input_ds[tz][ty][tx] = input[z_i * y_size * x_size + y_i * x_size + x_i];
    } else{
      input_ds[tz][ty][tx] = 0.0f;
    }
  __syncthreads();
  float out = 0.0f;
  if(tx < TILE_SIZE && ty < TILE_SIZE && tz < TILE_SIZE){
    for(int i = 0; i < KERNEL_WIDTH; i++){
      for(int j = 0; j < KERNEL_WIDTH; j++){
        for(int k = 0; k < KERNEL_WIDTH; k++){
          out += kernel[k][j][i] * input_ds[k + tz][j + ty][i + tx];
        }
      }
    }
    if(x < x_size && y < y_size && z < z_size){
      output[z * x_size * y_size + y * x_size + x] = out;
    }
  }

  // int idx = z * x_size * y_size + y * x_size + x;
  // float Pvalue = 0.0f;
  // if(x >= x_size || y >= y_size || z >= z_size){
  //   return;
  // }
  // for(int i = -1; i < 2; i++){
  //   for(int j = -1; j < 2; j++){
  //     for(int k = -1; k < 2; k ++){
  //       if((x + i < 0 || x + i >= x_size) 
  //       || (y + j < 0 || y + j >= y_size)
  //       || (z + k < 0 || z + k >= z_size)){
  //         continue;
  //       }
  //       Pvalue += input[(z + k) * x_size * y_size + (y + j) * x_size + x + i] * kernel[k + 1][j + 1][i + 1];
  //     }
  //   }
  // }
  // output[idx] = Pvalue;
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

  dim3 dimGrid(ceil((1.0 * x_size) / TILE_SIZE), ceil((1.0 * y_size) / TILE_SIZE), ceil((1.0 * z_size) / TILE_SIZE));
  dim3 dimBlock(TILE_SIZE + 2, TILE_SIZE + 2, TILE_SIZE + 2);
  
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
