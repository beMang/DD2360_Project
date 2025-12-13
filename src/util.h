#ifndef UTIL_H
#define UTIL_H

#include <vector>

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

float MSE_error(const char* file1, const char* file2){
    FILE* f1 = fopen(file1, "r");
    FILE* f2 = fopen(file2, "r");
    if (f1 == nullptr || f2 == nullptr) {
        printf("Error opening files for comparison.\n");
        return 0.0f;
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

    fclose(f1);
    fclose(f2);

    return mse;
}

float SSIM_error(const char* file1, const char* file2, int width, int height){
    std::vector<float> img1 = std::vector<float>();
    std::vector<float> img2 = std::vector<float>();

    FILE* f1 = fopen(file1, "r");
    FILE* f2 = fopen(file2, "r");
    if (f1 == nullptr || f2 == nullptr) {
        printf("Error opening files for comparison.\n");
        return 0.0f;
    }
    int skip_line = 3;
    //skip header and read pixel values
    char line1[256], line2[256];
    for (int i = 0; i < skip_line; i++) {
        [[maybe_unused]] char* ret1 = fgets(line1, sizeof(line1), f1);
        [[maybe_unused]] char* ret2 = fgets(line2, sizeof(line2), f2);
    }

    while (fgets(line1, sizeof(line1), f1) && fgets(line2, sizeof(line2), f2)) {
        int r1, g1, b1;
        int r2, g2, b2;
        sscanf(line1, "%d %d %d", &r1, &g1, &b1);
        sscanf(line2, "%d %d %d", &r2, &g2, &b2);
        img1.push_back(r1 / 255.0f);
        img1.push_back(g1 / 255.0f);
        img1.push_back(b1 / 255.0f);
        img2.push_back(r2 / 255.0f);
        img2.push_back(g2 / 255.0f);
        img2.push_back(b2 / 255.0f);
    }
    fclose(f1);
    fclose(f2);

    const int W = 8;
    const float K1 = 0.01f;
    const float K2 = 0.03f;
    const float L = 1.0f;
    const float C1 = (K1 * L) * (K1 * L);
    const float C2 = (K2 * L) * (K2 * L);

    float ssim_sum = 0.0f;
    int count = 0;

    for (int y = 0; y + W <= height; y++) {
        for (int x = 0; x + W <= width; x++) {

            float mu_x = 0, mu_y = 0;
            for (int j = 0; j < W; j++)
                for (int i = 0; i < W; i++) {
                    int idx = (y+j)*width + (x+i);
                    mu_x += img1[idx];
                    mu_y += img2[idx];
                }

            mu_x /= (W*W);
            mu_y /= (W*W);

            float sigma_x = 0, sigma_y = 0, sigma_xy = 0;
            for (int j = 0; j < W; j++)
                for (int i = 0; i < W; i++) {
                    int idx = (y+j)*width + (x+i);
                    float dx = img1[idx] - mu_x;
                    float dy = img2[idx] - mu_y;
                    sigma_x += dx * dx;
                    sigma_y += dy * dy;
                    sigma_xy += dx * dy;
                }

            sigma_x /= (W*W - 1);
            sigma_y /= (W*W - 1);
            sigma_xy /= (W*W - 1);

            float num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2);
            float den = (mu_x*mu_x + mu_y*mu_y + C1) *
                        (sigma_x + sigma_y + C2);

            ssim_sum += num / den;
            count++;
        }
    }
    return ssim_sum / count;
}