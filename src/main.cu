#include <iostream>
#include <time.h>
#include <vector>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "bvh.h"
#include "camera.h"
#include "material.h"
#include "util.h"
#include "kernel.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

int main(int argc, char** argv) {
    int nx = 1200;
    int ny = 800;
    int ns = 10;
    int tx = 8;
    int ty = 8;
    int n_obj = 22*22 + 4;

    if (argc > 1) n_obj = atoi(argv[1]);
    if (argc > 2) ns = atoi(argv[2]);

    std::cout << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel\n";
    std::cout << "in " << tx << "x" << ty << " blocks.\n";

    clock_t start, stop;
    start = clock();

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3_8bit);

    // allocate FB
    vec3_8bit *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state for per-pixel sampling
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    // Compute grid size needed for generating scene
    int grid_size = static_cast<int>(sqrtf(static_cast<float>(n_obj - 4)));
    int gen_block_size = 32;
    n_obj = grid_size * grid_size + 4; // adjust n_obj to match grid

    // Build scene on GPU using original curand sequence, then pull to host for BVH build
    hitable **d_hitable;
    checkCudaErrors(cudaMalloc((void**)&d_hitable, n_obj * sizeof(hitable*)));
    generate_scene_data<<<dim3(grid_size/gen_block_size+1, grid_size/gen_block_size+1),dim3(gen_block_size,gen_block_size)>>>(d_hitable, grid_size);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("Generated %d spheres.\n", n_obj);

    std::vector<int> prim_indices(n_obj);
    std::vector<aabb> prim_boxes(n_obj);
    for (int i = 0; i < n_obj; i++) prim_indices[i] = i;

    // Compute primitive bounding boxes on device and copy to host
    int tpb_bb = 128;
    int bpg_bb = (n_obj + tpb_bb - 1) / tpb_bb;
    aabb *d_prim_boxes = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_prim_boxes, n_obj * sizeof(aabb)));
    compute_bounding_boxes<<<bpg_bb, tpb_bb>>>(d_hitable, n_obj, d_prim_boxes);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy bounding boxes to host
    checkCudaErrors(cudaMemcpy(prim_boxes.data(), d_prim_boxes, n_obj * sizeof(aabb), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_prim_boxes));

    // Compute BVH on host
    std::vector<BVHNodeData> nodes;
    nodes.reserve(2 * n_obj);
    build_sah_bvh(prim_indices, 0, n_obj, prim_boxes, nodes, 4);

    // Reorder sphere array on GPU to match the final BVH leaf ordering
    int* d_prim_indices;
    checkCudaErrors(cudaMalloc((void**)&d_prim_indices, n_obj * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_prim_indices, prim_indices.data(), n_obj * sizeof(int), cudaMemcpyHostToDevice));
    
    hitable** d_hitable_reordered;
    checkCudaErrors(cudaMalloc((void**)&d_hitable_reordered, n_obj * sizeof(hitable*)));
    
    reorder_hitables<<<bpg_bb, tpb_bb>>>(d_hitable, d_hitable_reordered, d_prim_indices, n_obj);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cudaFree(d_hitable));
    checkCudaErrors(cudaFree(d_prim_indices));
    d_hitable = d_hitable_reordered;

    // Copy BVH nodes to device
    BVHNodeData *d_nodes;
    checkCudaErrors(cudaMalloc((void**)&d_nodes, nodes.size() * sizeof(BVHNodeData)));
    checkCudaErrors(cudaMemcpy(d_nodes, nodes.data(), nodes.size() * sizeof(BVHNodeData), cudaMemcpyHostToDevice));

    // World and camera
    hitable **d_world;
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world_from_flat<<<1,1>>>(d_nodes, static_cast<int>(nodes.size()), d_hitable, d_world, d_camera, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cout << "took " << timer_seconds << " seconds with " << n_obj << " objects.\n";

    // output FB as ppm image in file
    std::string output_image_filename = "tmp/image.ppm";
    saveFramebufferAsPPM(output_image_filename.c_str(), fb, nx, ny);

    // Compute and prints metrics
    std::string ref_image_filename = "tmp/ref.ppm";
    float mse = MSE_error(output_image_filename.c_str(), ref_image_filename.c_str());
    printf("Mean Squared Error (MSE) between frames: %f %%\n", 100*mse);
    double psnr = 10.0 * log10(1.0 / mse);
    printf("Peak Signal-to-Noise Ratio (PSNR): %f dB\n", psnr);

    float ssim = SSIM_error(output_image_filename.c_str(), ref_image_filename.c_str(), nx, ny);
    printf("Structural Similarity Index (SSIM) between frames: %f %%\n", 100*ssim);

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<bpg_bb, tpb_bb>>>(d_hitable, n_obj, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_hitable));
    checkCudaErrors(cudaFree(d_nodes));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}
