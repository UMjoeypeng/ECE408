// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void copy(float *input, float *buffer, int len){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) {buffer[i] = input[i];}
}

__global__ void total_whole(float *input, float *buffer, int len, int stride){
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(blockIdx.x >= stride && i < len) input[i] += buffer[(blockIdx.x - stride) * BLOCK_SIZE + BLOCK_SIZE - 1];
  // if(blockIdx.x >= stride && i < len) input[i] += ps[blockIdx.x - stride];
  __syncthreads();
  if(i < len) {buffer[i] = input[i];}
}


__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float XY[BLOCK_SIZE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    XY[threadIdx.x] = input[i]; 
  }
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if (threadIdx.x >= stride) XY[threadIdx.x] += XY[threadIdx.x - stride];
  }

  output[i] = XY[threadIdx.x];

}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceBuffer;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceBuffer, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<ceil(numElements / (float)BLOCK_SIZE), BLOCK_SIZE>>>(deviceInput, deviceOutput, numElements);
  cudaDeviceSynchronize();
  copy<<<ceil(numElements / (float)BLOCK_SIZE), BLOCK_SIZE>>>(deviceOutput, deviceBuffer, numElements);
  cudaDeviceSynchronize();
  for(int stride = 1; stride < ceil(numElements / (float)BLOCK_SIZE); stride *= 2){
    total_whole<<<ceil(numElements / (float)BLOCK_SIZE), BLOCK_SIZE>>>(deviceOutput, deviceBuffer, numElements, stride);
  }
  // total_scan<<<ceil(numElements / (float)BLOCK_SIZE), BLOCK_SIZE>>>(deviceOutput, deviceBuffer, numElements);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceBuffer);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
