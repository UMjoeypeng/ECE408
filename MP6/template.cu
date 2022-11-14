// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

#define BLOCK_SIZE 16
//@@ insert code here


__global__ void greyScale(float *deviceInput, unsigned char* deviceBuffer, unsigned char* deviceGrey, int width, int height, int channels){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * BLOCK_SIZE + tx;
  int y = blockIdx.y * BLOCK_SIZE + ty;

  int i = x * height + y;

  if(x < width && y < height){
    unsigned char r  = (unsigned char) (255 * deviceInput[3*i]);
    unsigned char g = (unsigned char) (255 * deviceInput[3*i + 1]);
    unsigned char b = (unsigned char) (255 * deviceInput[3*i + 2]);
    deviceBuffer[3 * i] = r;
    deviceBuffer[3 * i + 1] = g;
    deviceBuffer[3 * i + 2] = b;
    deviceGrey[i] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
  __syncthreads();
}

__global__ void computeHistogram(unsigned char* deviceGrey, int* histogram, int width, int height){
  int tx = threadIdx.x;
  int x = blockIdx.x * BLOCK_SIZE + tx;
  // int stride = width;
  for(int y = 0; y < height; y++){
    int i = x * height + y;
    if(x < width && y < height){
      atomicAdd(&(histogram[deviceGrey[i]]), 1);
    }
  }
}

__global__ void correct_color(unsigned char* deviceBuffer, float *deviceOutput, float *cdf, float cdfmin, int width, int height){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * BLOCK_SIZE + tx;
  int y = blockIdx.y * BLOCK_SIZE + ty;

  int i = x * height + y;
  
  if(x < width && y < height){
    for(int c = 0; c < 3; c++){
      int idx = 3 * i + c;
      float new_color = 255.0*(cdf[deviceBuffer[idx]] - cdfmin)/(1.0 - cdfmin);
      new_color = max(new_color, 0.0);
      new_color = min(new_color, 255.0);
      deviceOutput[idx] = (float)(new_color/255.0);
    }
  }
  __syncthreads();
}

float p(int x, int size){
  return (float) x / (float) size;
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInput;
  unsigned char* deviceBuffer;
  unsigned char* deviceGrey;
  float *deviceOutput;
  int *histogram;
  int *host_histogram;
  float *cdf;
  float *host_cdf;
  
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  host_histogram = (int *) malloc(256 * sizeof(int));
  host_cdf = (float *) malloc(256 * sizeof(float));


  
  int sizeInput = imageWidth * imageHeight * imageChannels * sizeof(float);
  int sizeBuffer = imageWidth * imageHeight * imageChannels * sizeof(unsigned char);
  int sizeGrey = imageWidth * imageHeight * sizeof(unsigned char);
  int sizeHis = 256 * sizeof(int);
  int sizeCDF = 256 * sizeof(float);
  cudaMalloc((void **)&deviceInput, sizeInput);
  cudaMalloc((void **)&deviceBuffer, sizeBuffer);
  cudaMalloc((void **)&deviceGrey, sizeGrey);
  cudaMalloc((void **)&histogram, sizeHis);
  cudaMalloc((void **)&cdf, sizeCDF);
  cudaMalloc((void **)&deviceOutput, sizeInput);


  cudaMemcpy(deviceInput, hostInputImageData, sizeInput, cudaMemcpyHostToDevice);


  dim3 dimGrid(ceil((1.0 * imageWidth) / BLOCK_SIZE), ceil((1.0 * imageHeight) / BLOCK_SIZE), 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  greyScale<<<dimGrid, dimBlock>>>(deviceInput, deviceBuffer, deviceGrey, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  dim3 dimGrid2(ceil((1.0 * imageWidth) / BLOCK_SIZE), 1, 1);
  dim3 dimBlock2(BLOCK_SIZE, 1, 1);
  computeHistogram<<<dimGrid2, dimBlock2>>>(deviceGrey, histogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  cudaMemcpy(host_histogram, histogram, sizeHis, cudaMemcpyDeviceToHost);

  int imagesize = imageWidth * imageHeight;
  host_cdf[0] = p(host_histogram[0], imagesize);
  for (int i = 1; i < 256; i++){
    host_cdf[i] = host_cdf[i - 1] + p(host_histogram[i], imagesize);
  }
  // printf("%f,%f\n", host_cdf[0], host_cdf[255]);
  cudaMemcpy(cdf, host_cdf, sizeCDF, cudaMemcpyHostToDevice);

  // for (int i = 0; i < 256; i++){
  //   printf("%d:%d\n", i, host_histogram[i]);
  // }
  correct_color<<<dimGrid, dimBlock>>>(deviceBuffer, deviceOutput, cdf, host_cdf[0], imageWidth, imageHeight);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutput, sizeInput, cudaMemcpyDeviceToHost);


  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInput);
  cudaFree(deviceBuffer);
  cudaFree(deviceOutput);
  cudaFree(deviceGrey);
  cudaFree(histogram);
  cudaFree(cdf);

  free(host_histogram);
  free(host_cdf);
  return 0;
}
