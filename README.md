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
* Generate scene on GPU in parallel
* Profiling done for each part of program (for reference and optimized implementation)

## Idea
* Make the creation of the world parallel : might worth it for large amount of object
* Removal of virtual function
* Impact of unified memory

## Profiling results

### Profiling reference implementation
Profiling with 7748 objects :

Reference implementation :
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

BHV implementation (but world creation is not parallel):
```
Rendering a 1200x800 image with 10 samples per pixel
in 8x8 blocks.
==336549== NVPROF is profiling process 336549, command: ./bin/cudart 8000
Generated 7748 spheres.
took 3.22066 seconds with 7748 objects.
Mean Squared Error (MSE) between frames: 4.180320 %
Peak Signal-to-Noise Ratio (PSNR): 13.787904 dB
Structural Similarity Index (SSIM) between frames: 81.873306 %
==336549== Profiling application: ./bin/cudart 8000
==336549== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   92.30%  2.76365s         1  2.76365s  2.76365s  2.76365s  generate_scene_data(hitable**, int*)
                    7.62%  228.12ms         1  228.12ms  228.12ms  228.12ms  render(vec3_8bit*, int, int, int, camera**, hitable**, curandStateXORWOW*)
                    0.05%  1.3797ms         1  1.3797ms  1.3797ms  1.3797ms  free_world(hitable**, int, hitable**, camera**)
                    0.01%  322.45us         1  322.45us  322.45us  322.45us  create_camera_kernel(camera**, int, int)
                    0.01%  322.00us         1  322.00us  322.00us  322.00us  create_world_from_flat(BVHNodeData const *, int, hitable**, hitable**)
                    0.01%  298.26us         1  298.26us  298.26us  298.26us  render_init(int, int, curandStateXORWOW*)
                    0.00%  33.633us         2  16.816us     928ns  32.705us  [CUDA memcpy DtoH]
                    0.00%  31.172us         3  10.390us     577ns  27.522us  [CUDA memcpy HtoD]
                    0.00%  6.3360us         1  6.3360us  6.3360us  6.3360us  compute_bounding_boxes(hitable**, int, aabb*)
                    0.00%  1.5680us         1  1.5680us  1.5680us  1.5680us  reorder_hitables(hitable**, hitable**, int*, int)
      API calls:   93.22%  2.99415s         9  332.68ms  3.8370us  2.76365s  cudaDeviceSynchronize
                    4.81%  154.60ms         1  154.60ms  154.60ms  154.60ms  cudaMallocManaged
                    1.73%  55.581ms         1  55.581ms  55.581ms  55.581ms  cudaDeviceReset
                    0.16%  5.0304ms         8  628.79us  4.2950us  4.9262ms  cudaLaunchKernel
                    0.04%  1.4153ms         9  157.26us  2.5970us  1.2416ms  cudaFree
                    0.02%  489.82us         9  54.424us  3.2980us  286.12us  cudaMalloc
                    0.01%  285.53us       114  2.5040us     307ns  106.13us  cuDeviceGetAttribute
                    0.01%  184.98us         5  36.996us  20.332us  59.149us  cudaMemcpy
                    0.00%  27.740us         1  27.740us  27.740us  27.740us  cuDeviceGetName
                    0.00%  14.083us         1  14.083us  14.083us  14.083us  cuDeviceTotalMem
                    0.00%  7.3540us         1  7.3540us  7.3540us  7.3540us  cuDeviceGetPCIBusId
                    0.00%  3.0210us         3  1.0070us     422ns  2.0510us  cuDeviceGetCount
                    0.00%  2.9830us         8     372ns     115ns  1.5440us  cudaGetLastError
                    0.00%  1.7020us         2     851ns     357ns  1.3450us  cuDeviceGet
                    0.00%     761ns         1     761ns     761ns     761ns  cuModuleGetLoadingMode
                    0.00%     527ns         1     527ns     527ns     527ns  cuDeviceGetUuid

==336549== Unified Memory profiling result:
Device "NVIDIA GeForce GTX 1080 Ti (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      25  112.64KB  4.0000KB  0.9961MB  2.750000MB  258.0930us  Device To Host
      10         -         -         -           -  1.968011ms  Gpu page fault groups
Total CPU Page faults: 12
```

BHV implentation + parallel world creation :
```
Rendering a 1200x800 image with 10 samples per pixel
in 8x8 blocks.
==335948== NVPROF is profiling process 335948, command: ./bin/cudart 8000
Generated 7925 spheres.
took 0.511827 seconds with 7925 objects.
Mean Squared Error (MSE) between frames: 4.424985 %
Peak Signal-to-Noise Ratio (PSNR): 13.540882 dB
Structural Similarity Index (SSIM) between frames: 81.248840 %
==335948== Profiling application: ./bin/cudart 8000
==335948== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   97.77%  273.92ms         1  273.92ms  273.92ms  273.92ms  render(vec3_8bit*, int, int, int, camera**, hitable**, curandStateXORWOW*)
                    1.06%  2.9674ms         1  2.9674ms  2.9674ms  2.9674ms  generate_scene_data(hitable**, int)
                    0.66%  1.8544ms         1  1.8544ms  1.8544ms  1.8544ms  free_world(hitable**, int, hitable**, camera**)
                    0.38%  1.0564ms         1  1.0564ms  1.0564ms  1.0564ms  create_world_from_flat(BVHNodeData const *, int, hitable**, hitable**, camera**, int, int)
                    0.11%  307.25us         1  307.25us  307.25us  307.25us  render_init(int, int, curandStateXORWOW*)
                    0.01%  33.217us         2  16.608us  3.8080us  29.409us  [CUDA memcpy HtoD]
                    0.01%  23.297us         1  23.297us  23.297us  23.297us  [CUDA memcpy DtoH]
                    0.00%  7.9050us         1  7.9050us  7.9050us  7.9050us  compute_bounding_boxes(hitable**, int, aabb*)
                    0.00%  2.1760us         1  2.1760us  2.1760us  2.1760us  reorder_hitables(hitable**, hitable**, int*, int)
      API calls:   55.83%  280.19ms         8  35.023ms  3.1540us  273.93ms  cudaDeviceSynchronize
                   32.62%  163.68ms         1  163.68ms  163.68ms  163.68ms  cudaMallocManaged
                   10.04%  50.375ms         1  50.375ms  50.375ms  50.375ms  cudaDeviceReset
                    0.99%  4.9602ms         7  708.60us  4.8890us  4.8534ms  cudaLaunchKernel
                    0.31%  1.5705ms         9  174.51us  2.7710us  1.2835ms  cudaFree
                    0.10%  517.01us         8  64.625us  3.1240us  310.13us  cudaMalloc
                    0.07%  330.22us       114  2.8960us     348ns  118.94us  cuDeviceGetAttribute
                    0.03%  150.66us         3  50.219us  24.000us  94.890us  cudaMemcpy
                    0.01%  26.114us         1  26.114us  26.114us  26.114us  cuDeviceGetName
                    0.00%  17.219us         1  17.219us  17.219us  17.219us  cuDeviceTotalMem
                    0.00%  8.7720us         1  8.7720us  8.7720us  8.7720us  cuDeviceGetPCIBusId
                    0.00%  3.5220us         3  1.1740us     567ns  2.3000us  cuDeviceGetCount
                    0.00%  2.9970us         7     428ns     140ns  1.3530us  cudaGetLastError
                    0.00%  1.9590us         2     979ns     481ns  1.4780us  cuDeviceGet
                    0.00%  1.3850us         1  1.3850us  1.3850us  1.3850us  cuModuleGetLoadingMode
                    0.00%     636ns         1     636ns     636ns     636ns  cuDeviceGetUuid

==335948== Unified Memory profiling result:
Device "NVIDIA GeForce GTX 1080 Ti (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      25  112.64KB  4.0000KB  0.9961MB  2.750000MB  257.4520us  Device To Host
      12         -         -         -           -  2.006379ms  Gpu page fault groups
Total CPU Page faults: 12
```