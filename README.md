# DD2360 Project

This repo contains the project for DD2360 applied GPU programming course at KTH. It implements a simple ray tracer, with the [proposed implementation](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/). Our goal is to optimise the project to render an image faster.

## Contributors
* Adrien Antonutti
* Kei Duke-Bergman
* Giovanni Prete

## Image file
Image are exported in PPM format, `ref.ppm` contains the original image of proposed implementation. It is kept for comparison purpose. `image.ppm` is obtained with our implementation.

## Adrien's modification done :
* Export PPM file directly instead of redirecting stdout
* Round to 8 bits before copy back to CPU (image is little different, but 8bits vary only form 1 so negligible (can by exactly the same if 25.99 is a double))
* Implement SSIM and MSE metrics for comparison
* BVH implementation : this drastically reduce the number of hit done. Works better for large amount of object.
* Input argument : number of object (this is not precise due to the way it was done in reference implementation) and number of sample per pixel

## Idea
* Make the creation of the world parallel : might worth it for large amount of object
* Removal of virtual function
* Impact of unified memory

## Profiling results

### Profiling reference implementation
Profiling with 7748 objects :

```
Rendering a 1200x800 image with 10 samples per pixel in 8x8 blocks.
==329503== NVPROF is profiling process 329503, command: ./bin/cudart_ref 8000
rand_init: 0.183724 sec
create_world: 2.812673 sec
render: 30.685343 sec
took 33.6818 for image gen. seconds with 7748 objects.
save_image: 0.157022 sec
free_cuda: 0.387908 sec
==329503== Profiling application: ./bin/cudart_ref 8000
==329503== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   90.71%  30.5301s         1  30.5301s  30.5301s  30.5301s  render(vec3*, int, int, int, camera**, hitable**, curandStateXORWOW*)
                    8.31%  2.79635s         1  2.79635s  2.79635s  2.79635s  create_world(hitable**, hitable**, camera**, int, int, curandStateXORWOW*, int*)
                    0.98%  331.10ms         1  331.10ms  331.10ms  331.10ms  free_world(hitable**, hitable**, camera**)
                    0.00%  299.06us         1  299.06us  299.06us  299.06us  render_init(int, int, curandStateXORWOW*)
                    0.00%  2.6880us         1  2.6880us  2.6880us  2.6880us  rand_init(curandStateXORWOW*)
                    0.00%     993ns         1     993ns     993ns     993ns  [CUDA memcpy DtoH]
                    0.00%     288ns         1     288ns     288ns     288ns  [CUDA memcpy HtoD]
      API calls:   98.41%  33.3268s         5  6.66535s  8.3850us  30.5301s  cudaDeviceSynchronize
                    0.98%  332.90ms         7  47.558ms  3.5130us  331.11ms  cudaFree
                    0.42%  142.68ms         1  142.68ms  142.68ms  142.68ms  cudaMallocManaged
                    0.16%  55.615ms         1  55.615ms  55.615ms  55.615ms  cudaDeviceReset
                    0.01%  4.7750ms         5  955.00us  5.8770us  4.3435ms  cudaLaunchKernel
                    0.00%  521.58us         6  86.930us  5.9250us  342.54us  cudaMalloc
                    0.00%  259.72us       114  2.2780us     282ns  100.45us  cuDeviceGetAttribute
                    0.00%  57.728us         2  28.864us  24.038us  33.690us  cudaMemcpy
                    0.00%  27.847us         1  27.847us  27.847us  27.847us  cuDeviceGetName
                    0.00%  14.565us         1  14.565us  14.565us  14.565us  cuDeviceTotalMem
                    0.00%  7.8890us         1  7.8890us  7.8890us  7.8890us  cuDeviceGetPCIBusId
                    0.00%  3.3680us         5     673ns     121ns  1.2500us  cudaGetLastError
                    0.00%  2.8030us         3     934ns     420ns  1.8900us  cuDeviceGetCount
                    0.00%  1.5490us         1  1.5490us  1.5490us  1.5490us  cuModuleGetLoadingMode
                    0.00%  1.5010us         2     750ns     356ns  1.1450us  cuDeviceGet
                    0.00%     511ns         1     511ns     511ns     511ns  cuDeviceGetUuid

==329503== Unified Memory profiling result:
Device "NVIDIA GeForce GTX 1080 Ti (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      96  117.21KB  4.0000KB  0.9961MB  10.98828MB  947.1550us  Device To Host
      33         -         -         -           -  5.294023ms  Gpu page fault groups
Total CPU Page faults: 35
```