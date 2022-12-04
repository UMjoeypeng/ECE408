#include <cmath>
#include <iostream>

#include "gpu-new-forward.h"

#define TILE_WIDTH 4
// #define TILE_SIZE 256

__global__ void conv_forward_kernel(float *output, const float *input,
                                    const float *mask, const int Batch,
                                    const int Map_out, const int Channel,
                                    const int Height, const int Width,
                                    const int K, int tile_size) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire
    mini-batch The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning.
    // remove this line when you start working (void)Width_out; // silence
    // declared but never referenced warning. remove this line when you start
    // working

    // We have some nice #defs for you below to simplify indexing. Feel free to
    // use them, or create your own. An example use of these macros: float a =
    // in_4d(0,0,0,0) out_4d(0,0,0,0) = a

#define out_4d(i3, i2, i1, i0)                         \
    output[(i3) * (Map_out * Height_out * Width_out) + \
           (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
#define in_4d(i3, i2, i1, i0)                                           \
    input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + \
          (i1) * (Width) + i0]
#define mask_4d(i3, i2, i1, i0) \
    mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_out = Width - K + 1;
    int H_out = Height - K + 1;
    __shared__ float input_tile[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];
    __shared__ float mask_tile[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];

    int unroll_mask_width = Channel * K * K;
    int unroll_mask_height = Map_out;
    int unroll_input_width = H_out * W_out;
    int unroll_input_height = Channel * K * K;

    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

    int x = blockIdx.x * tile_size + tx; // width & height
    int y = blockIdx.y * tile_size + ty; // Output channel
    int z = blockIdx.z * tile_size + tz; // Batch

    int h_out = x / W_out;  // Which pixel in the output image
    int w_out = x % W_out;

                            
    float ans = 0.0;
    

    if(z < Batch){
        for(int i = 0; i < ceil((float)unroll_mask_width / (float)tile_size); i++){
            int tile_x = i * tile_size + tx;
            int tile_y = i * tile_size + ty;
            int p = (tile_y % (K*K)) / K;
            int q = (tile_y % (K*K)) % K;

            if(y < unroll_mask_height && tile_y < unroll_mask_height){
                mask_tile[tz][ty][tx] = mask[y*Channel * K * K+tile_x];
            } else {
                mask_tile[tz][ty][tx] = 0;
            }

            if(tile_y < unroll_input_height && x < unroll_input_width){
                input_tile[tz][ty][tx] = in_4d(z, tile_y / (K * K), h_out + p, w_out + q);
            } else {
                input_tile[tz][ty][tx] = 0;
            }

            __syncthreads();

            for(int k = 0; k< tile_size;k++){
                ans += mask_tile[tz][ty][k] * input_tile[tz][k][tx];
            }
            __syncthreads();
        }
        if(x < unroll_input_width && y < unroll_mask_height){
            output[z * unroll_input_width * unroll_mask_height + y * unroll_input_width + x] = ans;
        }
    }
                 

#undef out_4d
#undef in_4d
#undef mask_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(
    const float *host_output, const float *host_input, const float *host_mask,
    float **device_output_ptr, float **device_input_ptr,
    float **device_mask_ptr, const int Batch, const int Map_out,
    const int Channel, const int Height, const int Width, const int K) {
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device
    // pointers,
    //  which are passed to the other two functions.

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int size_input = Batch * Channel * Height * Width * sizeof(float);
    int size_mask = Channel * Map_out * K * K * sizeof(float);
    int size_output = Batch * Map_out * Height_out * Width_out * sizeof(float);
    cudaMalloc((void **)device_input_ptr, size_input);
    cudaMalloc((void **)device_output_ptr, size_output);
    cudaMalloc((void **)device_mask_ptr, size_mask);

    cudaMemcpy(*device_input_ptr, host_input, size_input,
               cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, size_mask, cudaMemcpyHostToDevice);
    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}

__host__ void GPUInterface::conv_forward_gpu(
    float *device_output, const float *device_input, const float *device_mask,
    const int Batch, const int Map_out, const int Channel, const int Height,
    const int Width, const int K) {
    // Set the kernel dimensions and call the kernel
    int W_out = Width - K + 1;
    int H_out = Height - K + 1;

    // int unroll_mask_width = Channel * K * K;
    int unroll_mask_height = Map_out;
    int unroll_input_width = H_out * W_out;
    // int unroll_input_height = Channel * K * K;
    // int tile_size = Map_out > 4 ? 8 : 4;
    int tile_size = 4;
    dim3 blockDim(tile_size, tile_size, tile_size);
    dim3 gridDim(ceil((1.0 * unroll_input_width) / tile_size), ceil((1.0 * unroll_mask_height) / tile_size), ceil((1.0 * Batch)/ tile_size));

    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K, tile_size);
    


    // int W_grid = ceil(W_out / (float)TILE_WIDTH);
    // int H_grid = ceil(H_out / (float)TILE_WIDTH);
    // int Z = H_grid * W_grid;
    // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 gridDim(Batch, Map_out, Z);
    // size_t shmem_size =
    //     sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);
    // conv_forward_kernel<<<gridDim, blockDim, shmem_size>>>(
    //     device_output, device_input, device_mask, Batch, Map_out, Channel,
    //     Height, Width, K);
    cudaDeviceSynchronize();
}

__host__ void GPUInterface::conv_forward_gpu_epilog(
    float *host_output, float *device_output, float *device_input,
    float *device_mask, const int Batch, const int Map_out, const int Channel,
    const int Height, const int Width, const int K) {
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMemcpy(host_output, device_output,
               (Batch * Map_out * Height_out * Width_out) * sizeof(float),
               cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}

__host__ void GPUInterface::get_device_properties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name
                  << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "."
                  << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem
                  << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem
                  << std::endl;
        std::cout << "Max Shared memory size per block: "
                  << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock
                  << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0]
                  << " x, " << deviceProp.maxThreadsDim[1] << " y, "
                  << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0]
                  << " x, " << deviceProp.maxGridSize[1] << " y, "
                  << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}
