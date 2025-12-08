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

void MSE_error(const char* file1, const char* file2){
    FILE* f1 = fopen(file1, "r");
    FILE* f2 = fopen(file2, "r");
    if (f1 == nullptr || f2 == nullptr) {
        printf("Error opening files for comparison.\n");
        return;
    }

    int skip_line = 3;

    //skip header and compute MSE 
    char line1[256], line2[256];
    for (int i = 0; i < skip_line; i++) {
        [[maybe_unused]] char* ret1 = fgets(line1, sizeof(line1), f1);
        [[maybe_unused]] char* ret2 = fgets(line2, sizeof(line2), f2);
    }

    double mse = 0.0;
    int count = 0;
    while (fgets(line1, sizeof(line1), f1) && fgets(line2, sizeof(line2), f2)) {
        int r1, g1, b1;
        int r2, g2, b2;
        sscanf(line1, "%d %d %d", &r1, &g1, &b1);
        sscanf(line2, "%d %d %d", &r2, &g2, &b2);
        mse += (r1 - r2) * (r1 - r2)/65025.0;
        mse += (g1 - g2) * (g1 - g2)/65025.0;
        mse += (b1 - b2) * (b1 - b2)/65025.0;
        count += 3; //Normalize by number of color channels
    }
    mse /= count;

    printf("Mean Squared Error (MSE) between frames: %f %%\n", 100*mse);
    double psnr = 10.0 * log10(1.0 / mse);
    printf("Peak Signal-to-Noise Ratio (PSNR): %f dB\n", psnr);

    fclose(f1);
    fclose(f2);
}