#include <iostream>
#include <time.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <random>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "bvh.h"
#include "camera.h"
#include "material.h"
#include "util.h"

__global__ void compute_bounding_boxes(hitable** d_list, int n, aabb* out_boxes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out_boxes[idx] = d_list[idx]->bounding_box();
    }
}

__global__ void reorder_hitables(hitable** src, hitable** dst, int* indices, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        dst[i] = src[indices[i]];
    }
}

__global__ void generate_scene_data(hitable** d_list, int grid_size) {
    int xid = threadIdx.x + blockIdx.x * blockDim.x;
    int yid = threadIdx.y + blockIdx.y * blockDim.y;

    if (xid == 0 && yid == 0) {
        d_list[0] = new sphere(vec3(0,-1000.0f,-1), 1000.0f, new lambertian(vec3(0.5f, 0.5f, 0.5f))); // ground
        //3 bigger sphere
        d_list[1] = new sphere(vec3(0,1,0), 1.0f, new dielectric(1.5f));
        d_list[2] = new sphere(vec3(-4,1,0), 1.0f, new lambertian(vec3(0.4f,0.2f,0.1f)));
        d_list[3] = new sphere(vec3(4,1,0), 1.0f, new metal(vec3(0.7f,0.6f,0.5f), 0.0f));
    }

    if(xid < grid_size && yid < grid_size) {
        int half_grid = grid_size/2;
        int id = xid + yid*grid_size +4;

        curandState rand_state;
        curand_init(1984 + id, 0, 0, &rand_state);
        
        float choose_mat = curand_uniform(&rand_state);
        vec3 center(xid - half_grid + curand_uniform(&rand_state), 0.2f, yid - half_grid + curand_uniform(&rand_state));
        if(choose_mat < 0.8f) {
            vec3 albedo(curand_uniform(&rand_state)*curand_uniform(&rand_state),
                        curand_uniform(&rand_state)*curand_uniform(&rand_state),
                        curand_uniform(&rand_state)*curand_uniform(&rand_state));
            d_list[id] = new sphere(center, 0.2f, new lambertian(albedo));
        }
        else if(choose_mat < 0.95f) {
            vec3 albedo(0.5f*(1.0f+curand_uniform(&rand_state)),
                        0.5f*(1.0f+curand_uniform(&rand_state)),
                        0.5f*(1.0f+curand_uniform(&rand_state)));
            float fuzz = 0.5f * curand_uniform(&rand_state);
            d_list[id] = new sphere(center, 0.2f, new metal(albedo, fuzz));
        }
        else {
            d_list[id] = new sphere(center, 0.2f, new dielectric(1.5f));
        }
    }
}
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

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3_8bit *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = vec3_8bit(static_cast<u_int8_t>(255.99f*col[0]), static_cast<u_int8_t>(255.99f*col[1]), static_cast<u_int8_t>(255.99f*col[2]));
}

__global__ void create_world_from_flat(const BVHNodeData* nodes, int node_count, hitable **d_list, hitable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_world = new bvh_flat_world(nodes, node_count, d_list);
    }
}

__global__ void create_camera_kernel(camera **d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.1f;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0f,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, int count, hitable **d_world, camera **d_camera) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        delete ((sphere *)d_list[idx])->mat_ptr;
        delete d_list[idx];
    }
    if (idx == 0 && blockIdx.x == 0) {
        delete *d_world;
        delete *d_camera;
    }
}

//number of objects in arguments
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
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world_from_flat<<<1,1>>>(d_nodes, static_cast<int>(nodes.size()), d_hitable, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    create_camera_kernel<<<1,1>>>(d_camera, nx, ny);
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
    //use string for filename
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
