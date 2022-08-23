#include "cuda.cuh"

#include <cstring>
#include "cuda_runtime.h"

#include "helper.h"
#include <stdlib.h>
#include <stdio.h>
#include "device_launch_parameters.h"


///
/// Algorithm storage
///
// Host copy of input image
Image cuda_input_image;
Image cuda_output_image;
// Host copy of image tiles in each dimension
unsigned int cuda_TILES_X, cuda_TILES_Y;
// Host copy of image height
unsigned int h_image_width;
// Host copy of image width
unsigned int h_image_height;
// Host copy of image channels
unsigned int h_image_channels;
// Pointer to host buffer for global pixel average sum
unsigned long long* h_global_pixel_sum;
// Pointer to device buffer for calculating the sum of each tile mosaic, this must be passed to a kernel to be used on device
unsigned char* h_mosaic_value;

// Pointer to device buffer for storing the output pixels of each tile, this must be passed to a kernel to be used on device
unsigned long long* d_mosaic_sum;
// Pointer to host buffer to store the mosaic value
unsigned char* d_mosaic_value;
// Pointer to device image data buffer, for storing the input image, this must be passed to a kernel to be used on device
unsigned char* d_input_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;
// Pointer to device buffer for the global pixel average sum, this must be passed to a kernel to be used on device
unsigned long long* d_global_pixel_sum;



//Pointer to device tile indices
unsigned long long* d_tile_indices;
//Pointer to tile offsets
unsigned long long* d_tile_offsets;
//Pointer to host mosaic sum
unsigned long long* h_mosaic_sum;
//Pointer to host tile offsets
unsigned long long* h_tile_offsets;

///Device constants
__constant__ int d_image_channels;
__constant__ int d_image_width;
__constant__ int d_tiles_x;
__constant__ int d_tiles_y;

//__constant__ unsigned char* d_input_image_data;


void cuda_begin(const Image *input_image) {
    // These are suggested CUDA memory allocations that match the CPU implementation
    // If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference

    cuda_TILES_X = input_image->width / TILE_SIZE;
    cuda_TILES_Y = input_image->height / TILE_SIZE;
    h_image_width = input_image->width;
    h_image_height = input_image->height;
    h_image_channels = input_image->channels;

    // Allocate buffer for calculating the sum of each tile mosaic
    CUDA_CALL(cudaMalloc(&d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long)));

    // Allocate buffer for storing the output pixel value of each tile
    CUDA_CALL(cudaMalloc(&d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned char)));


    const size_t image_data_size = input_image->width * input_image->height * input_image->channels * sizeof(unsigned char);
    // Allocate copy of input image
    cuda_input_image = *input_image;
    cuda_input_image.data = (unsigned char*)malloc(image_data_size);
    h_mosaic_sum = (unsigned long long*)malloc(cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long));
    h_mosaic_value = (unsigned char*)malloc(cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned char));
    h_global_pixel_sum = (unsigned long long*)malloc(input_image->channels * sizeof(unsigned long long));
    memcpy(cuda_input_image.data, input_image->data, image_data_size);



    // Allocate and fill device buffer for storing input image data
    CUDA_CALL(cudaMalloc(&d_input_image_data, image_data_size));
    CUDA_CALL(cudaMemcpy(d_input_image_data, input_image->data, image_data_size, cudaMemcpyHostToDevice));
    //CUDA_CALL(cudaMemcpyToSymbol(d_input_image_data, input_image->data, image_data_size));

    //Allocate device buffer for storing output image data
    CUDA_CALL(cudaMalloc(&d_output_image_data, image_data_size));

    //Allocate and zero buffer for calculation global pixel average
    CUDA_CALL(cudaMalloc(&d_global_pixel_sum, input_image->channels * sizeof(unsigned long long)));

    // Allocate and zero buffer for calculation tiles offsets
    CUDA_CALL(cudaMalloc(&d_tile_offsets, cuda_TILES_X + cuda_TILES_Y));

    // Copy image width device constant memory
    CUDA_CALL(cudaMemcpyToSymbol(d_image_width, &input_image->width, sizeof(int)));

    // Copy image channels to device constant memory
    CUDA_CALL(cudaMemcpyToSymbol(d_image_channels, &input_image->channels, sizeof(int)));

    // Copy X tiles to device constant memory
    CUDA_CALL(cudaMemcpyToSymbol(d_tiles_x, &cuda_TILES_X, sizeof(int)));

    // Copy Y tiles to device constant memory
    CUDA_CALL(cudaMemcpyToSymbol(d_tiles_y, &cuda_TILES_Y, sizeof(int)));

    // Allocate buffer to calculate global output average
    //CUDA_CALL(cudaMalloc(&d_output_global_average, input_image->channels * sizeof(char)));

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    for(int dev = 0; dev < deviceCount; ++dev){
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
    }
}


__global__ void mosaicSum(unsigned long long* d_mosaic_sum, unsigned char* d_input_image_data ) {
    //const int stride = TILE_PIXELS + 1;

    //__shared__ int tile_pixels[stride];


    //Each block is a tile
    int t_x = blockIdx.x;
    int t_y = blockIdx.y;

    //Each thread is a pixel
    int p_x = threadIdx.x;
    int p_y = threadIdx.y;
    //int ch = blockIdx.z;

    const unsigned int tile_index = (t_y * d_tiles_x + t_x) * d_image_channels;
    const unsigned int tile_offset = (t_y * d_tiles_x * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * d_image_channels;
    const unsigned int pixel_offset = (p_y * d_image_width + p_x) * d_image_channels;

    for (int ch = 0; ch < d_image_channels; ++ch) {
        unsigned int pixel = d_input_image_data[tile_offset + pixel_offset + ch];

        for (int offset = 16; offset > 0; offset /= 2)
            pixel += __shfl_up(pixel, offset);
    
        if (threadIdx.x % 32 == 0)
            atomicAdd(&d_mosaic_sum[tile_index + ch], pixel);

    }
    
    //9 whole milliseconds
    //atomicAdd(&d_mosaic_sum[tile_index + ch], pixel);
}


void cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_tile_sum(&cuda_input_image, h_mosaic_sum);
    // blocks = tiles, tilesize
    dim3 blocks(cuda_TILES_X, cuda_TILES_Y, 1);
    dim3 threads(h_image_width/cuda_TILES_X, h_image_height/cuda_TILES_Y, 1);
    mosaicSum<<<blocks, threads>>>(d_mosaic_sum, d_input_image_data);



#ifdef VALIDATION

    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    CUDA_CALL(cudaMemcpy(h_mosaic_sum, d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    validate_tile_sum(&cuda_input_image, h_mosaic_sum);
#endif
}

__global__ void eachTileSum(unsigned char* d_mosaic_value, unsigned long long* d_mosaic_sum, unsigned long long* d_global_pixel_sum) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    for (int ch = 0; ch < d_image_channels; ++ch) {
        unsigned char sum = d_mosaic_sum[t * d_image_channels + ch] / TILE_PIXELS;
        d_mosaic_value[t * d_image_channels + ch] = sum;
        atomicAdd(&d_global_pixel_sum[ch], int(sum));
    }
}


void cuda_stage2(unsigned char* output_global_average) {
    // Loading as many threads as needed
    // 
    unsigned int tilesN;
   if (cuda_TILES_X * cuda_TILES_Y > 1024){
       tilesN = 1024;
   }else{
       tilesN = cuda_TILES_Y * cuda_TILES_X;
   }
   // number of threads spread equally across blocks
   unsigned int blocksN = (cuda_TILES_X * cuda_TILES_Y) / tilesN;

    dim3 blocks(blocksN , cuda_input_image.channels, 1);
    dim3 threads(tilesN, 1, 1);

    eachTileSum<< <blocks, threads>> > (d_mosaic_value, d_mosaic_sum, d_global_pixel_sum);


#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // 
    //for (int ch = 0; ch < cuda_input_image.channels; ++ch) {
    //     output_global_average[ch] = (unsigned int)(global_pixel_sum[ch] / TILE_PIXELS);
    //printf("ch %d = %d", ch, output_global_average[ch]);
    //}
    // (Ensure that data copy is carried out within the ifdef
    //cudaMemcpy(global_pixel_sum, d_global_pixel_sum, cuda_input_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //cudaMemcpy(h_mosaic_sum, d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_mosaic_value, d_mosaic_value,  cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //cudaMemcpy(output_global_average, d_output_global_average, cuda_input_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //validate_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, h_mosaic_sum, h_mosaic_value, output_global_average);
#endif
}

__global__ void broadcastMosaic(unsigned char *d_output_image_data, unsigned char *d_mosaic_value) {

    unsigned int t_x = blockIdx.x;
    unsigned int t_y = blockIdx.y;

    unsigned int p_x = threadIdx.x;
    unsigned int p_y = threadIdx.y;
    unsigned int ch = blockIdx.z;

    
    const unsigned int tile_index = (t_y * d_tiles_x + t_x) * d_image_channels;
    const unsigned int tile_offset = (t_y * d_tiles_x * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * d_image_channels;
    const unsigned int pixel_offset = (p_y * d_image_width + p_x) * d_image_channels;
    
    
    //for(int ch =0; ch < d_image_channels; ++ch){
    //    d_output_image_data[tile_offset + pixel_offset + ch] = d_mosaic_value[tile_index + ch];
    //
    d_output_image_data[tile_offset + pixel_offset + ch] = d_mosaic_value[tile_index + ch];
}




void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(input_image, compact_mosaic, output_image);
    // Tile = block
    // Pixel = thread
    dim3 blocks(cuda_TILES_X, cuda_TILES_Y, 3);
    dim3 threads(TILE_SIZE, TILE_SIZE, 1);

    broadcastMosaic<<<blocks, threads>>>(d_output_image_data, d_mosaic_value);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    //CUDA_CALL(cudaMemcpy(h_mosaic_value, d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    //CUDA_CALL(cudaMemcpy(&cuda_output_image, d_output_image_data, cuda_input_image.width * cuda_input_image.height * cuda_input_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    //validate_broadcast(&cuda_input_image, h_mosaic_value, &cuda_output_image);
#endif
}
void cuda_end(Image *output_image) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    output_image->width = cuda_input_image.width;
    output_image->height = cuda_input_image.height;
    output_image->channels = cuda_input_image.channels;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    free(cuda_input_image.data);
    CUDA_CALL(cudaFree(d_mosaic_value));
    CUDA_CALL(cudaFree(d_mosaic_sum));
    CUDA_CALL(cudaFree(d_input_image_data));
    CUDA_CALL(cudaFree(d_output_image_data));
    CUDA_CALL(cudaFree(d_global_pixel_sum));
    CUDA_CALL(cudaFree(d_tile_offsets));
}


