#ifndef UTIL_H
#define UTIL_H

/**
 * Saves the framebuffer as a PPM image file.
 * @param filename The name of the output file.
 * @param framebuffer Pointer to the framebuffer array.
 * @param width Width of the image.
 * @param height Height of the image.
 */
void saveFramebufferAsPPM(const char* filename, vec3_8bit* framebuffer, int width, int height);
#endif // UTIL_H

#include <cstdio>
#include "vec3.h"

void saveFramebufferAsPPM(const char* filename, vec3_8bit* framebuffer, int width, int height) {
    FILE* file = fopen(filename, "w");
    if (file == nullptr) return;
    fprintf(file, "P3\n%d %d\n255\n", width, height);
    for (int j = height-1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j*width + i;
            int ir = framebuffer[pixel_index].e[0];
            int ig = framebuffer[pixel_index].e[1];
            int ib = framebuffer[pixel_index].e[2];
            fprintf(file, "%d %d %d\n", ir, ig, ib);
        }
    }
    fclose(file);
}