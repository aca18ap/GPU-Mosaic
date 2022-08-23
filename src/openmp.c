#include "openmp.h"
#include "helper.h"

#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stdio.h>

#define CPU_THREADS 8

Image openmp_input_image;
Image openmp_output_image;
unsigned int openmp_TILES_X, openmp_TILES_Y;
unsigned long long* openmp_mosaic_sum;
unsigned char* openmp_mosaic_value;
int threads_n;

void openmp_begin(const Image *input_image) {
    openmp_TILES_X = input_image->width / TILE_SIZE;
    openmp_TILES_Y = input_image->height / TILE_SIZE;
    // Allocate buffer for calculating the sum of each tile mosaic
    openmp_mosaic_sum = (unsigned long long*)malloc(openmp_TILES_X * openmp_TILES_Y * input_image->channels * sizeof(unsigned long long));

    // Allocate buffer for storing the output pixel value of each tile
    openmp_mosaic_value = (unsigned char*)malloc(openmp_TILES_X * openmp_TILES_Y * input_image->channels * sizeof(unsigned char));

    // Allocate copy of input image
    openmp_input_image = *input_image;
    openmp_input_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
    memcpy(openmp_input_image.data, input_image->data, input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));

    // Allocate output image
    openmp_output_image = *input_image;
    openmp_output_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));

    threads_n = omp_get_num_threads();

}

void openmp_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_tile_sum(&openmp_input_image, openmp_mosaic_sum);
    memset(openmp_mosaic_sum, 0, openmp_TILES_X * openmp_TILES_Y * openmp_input_image.channels * sizeof(unsigned long long));
#pragma omp parallel 
    {
    int t_x, t_y;
    
    #pragma omp for collapse(2)
        for (t_x = 0; t_x < openmp_TILES_X; ++t_x) {
            for (t_y = 0; t_y < openmp_TILES_Y; ++t_y) {
                //int thread = omp_get_thread_num();
                //int max_threads = omp_get_max_threads();
                //unsigned char tile_sum[4] = { 0,0,0,0 };
                const unsigned int tile_index = (t_y * openmp_TILES_X + t_x) * openmp_input_image.channels;
                //printf("Tile: %d - (Thread %d of %d)\n", tile_index, thread, max_threads);
                const unsigned int tile_offset = (t_y * openmp_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * openmp_input_image.channels;
                //fprintf(stderr,"tile index: %d, tile offset: %d\n", tile_index, tile_offset);
                int p_x, p_y, ch;
                // For each pixel within the tile

                for (p_x = 0; p_x < TILE_SIZE; ++p_x) {
                    for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
                        // For each colour channel
                        const unsigned int pixel_offset = (p_y * openmp_input_image.width + p_x) * openmp_input_image.channels;
                        for (ch = 0; ch < openmp_input_image.channels; ++ch) {
                            // Load pixel
                            const unsigned char pixel = openmp_input_image.data[tile_offset + pixel_offset + ch];
                            openmp_mosaic_sum[tile_index + ch] += pixel;
                        }
                    }
                }
            }
        }
    }



#ifdef VALIDATION
    validate_tile_sum(&openmp_input_image, openmp_mosaic_sum);
#endif
}

void openmp_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_compact_mosaic(TILES_X, TILES_Y, mosaic_sum, compact_mosaic, global_pixel_average);
    unsigned long long whole_image_sum[4] = { 0,0,0,0 };
    int t, ch;
    for (t = 0; t < openmp_TILES_X * openmp_TILES_Y; t++) {
        for (ch = 0; ch < openmp_input_image.channels; ++ch) {
            openmp_mosaic_value[t * openmp_input_image.channels + ch] = (unsigned char)(openmp_mosaic_sum[t * openmp_input_image.channels + ch] / TILE_PIXELS);
            whole_image_sum[ch] += openmp_mosaic_value[t * openmp_input_image.channels + ch];
        }
    }
    for (ch = 0; ch < openmp_input_image.channels; ++ch) {
        output_global_average[ch] = (unsigned char)(whole_image_sum[ch] / (openmp_TILES_X * openmp_TILES_Y));
    }

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    validate_compact_mosaic(openmp_TILES_X, openmp_TILES_Y, openmp_mosaic_sum, openmp_mosaic_value, output_global_average);
#endif    
}
void openmp_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_broadcast(openmp_input_image, compact_mosaic, openmp_output_image);
#pragma omp parallel
    {
        int t_x, t_y;
#pragma omp for collapse(2)
        for (t_x = 0; t_x < openmp_TILES_X; ++t_x) {
            for (t_y = 0; t_y < openmp_TILES_Y; ++t_y) {
                const unsigned int tile_index = (t_y * openmp_TILES_X + t_x) * openmp_input_image.channels;
                const unsigned int tile_offset = (t_y * openmp_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * openmp_input_image.channels;

                // For each pixel within the tile
                int p_x, p_y;
                for (p_x = 0; p_x < TILE_SIZE; ++p_x) {
                    for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
                        const unsigned int pixel_offset = (p_y * openmp_input_image.width + p_x) * openmp_input_image.channels;
                        // Copy whole pixel
                        memcpy(openmp_output_image.data + tile_offset + pixel_offset, openmp_mosaic_value + tile_index, openmp_input_image.channels);
                    }
                }
            }
        }
    }

    

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_broadcast(&openmp_input_image, openmp_mosaic_value, &openmp_output_image);
#endif    
}
void openmp_end(Image *output_image) {
    output_image->width = openmp_output_image.width;
    output_image->height = openmp_output_image.height;
    output_image->channels = openmp_output_image.channels;
    memcpy(output_image->data, openmp_output_image.data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char));
    // Release allocations
    free(openmp_output_image.data);
    free(openmp_input_image.data);
    free(openmp_mosaic_value);
    free(openmp_mosaic_sum);
}