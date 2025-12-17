#ifndef KERNELH
#define KERNELH

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

__global__ void create_world_from_flat(const BVHNodeData* nodes, int node_count, hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_world = new bvh_flat_world(nodes, node_count, d_list);

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

#endif