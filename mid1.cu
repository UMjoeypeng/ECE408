
#define TILE_WIDTH 16
// complete the host code here
__host__ void CrossSum(int *Data, int *OutData, int size) {
    int size_d = size*size*sizeof(int);
    float *device_d;
    float *device_o;
    cudaMalloc((void **)&device_d, size_d);
    cudaMalloc((void **)&device_o, size_d);
    cudaMemcpy(device_d, Data, size_d, cudaMemcpyHostToDevice);
    int block_width = TILE_WIDTH;
    dim3 dimGrid(ceil((1.0 * size) / block_width), ceil((1.0 * size) / block_width), 1);
    dim3 dimBlock(block_width, block_width, 1);

    rowcolSum<<<dimGrid, dimBlock>>>(device_d, device_o, size);

    cudaMemcpy(OutData, device_o, size_d, cudaMemcpyDeviceToHost);
    cudaFree(device_d);
    cudaFree(device_o);


}

// Write your CUDA kernel code here...

__global__ void rowcolSum(int* input, int* output, int size){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;
    for (int ph = 0; ph < ceil(numAColumns / (float)TILE_WIDTH); ++ph)
    {
        if ((Row < size) && (ph * TILE_WIDTH + tx) < size)
        Mds[ty][tx] = A[Row * numAColumns + ph * TILE_WIDTH + tx];
        else
        Mds[ty][tx] = 0;
        if ((ph * TILE_WIDTH + ty) < size && Col < size)
        Nds[ty][tx] = B[(ph * TILE_WIDTH + ty) * numBColumns + Col];
        else
        Nds[ty][tx] = 0;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)
        {
        Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if ((Row < size) && (Col < size))
        C[Row * size + Col] = Pvalue;
}