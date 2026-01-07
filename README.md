# DD2360 Project

This repo contains the project for DD2360 applied GPU programming course at KTH. It implements a simple ray tracer, with the [proposed implementation](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/). Our goal is to optimise the project to render an image faster.

## Modifications done

* Adrien Antonutti
   * 8 bits conversion for frame buffer
   * BVH implementation
   * Parallel scene generation and clean up
   * MSE and SSIM metrics for validation
* Kei Duke-Bergman
   * Impact of unified memory
   * Attempt memadvise and prefetch in Windows
* Giovanni Prete
   * Impact of removal of polymorphic class and function

## Project structure

* `bin` contains the binary of the project.
* `ref_src` contains the reference implementation : both the "real" reference and "parallelize scene creation" version that was used for validation of our optimised implementation. This has to be done because the random behavior of serial and parallel scene creation couldn't be exactly the same.
* `script` contains some python script to generate the graphs (those might be outdated)
* `src` contains our optimised implementation : both with and without virtual function (without being the most performant).
* `src_unified` contains the code related to unified memory experience (This is not compiled in the makefile but still provided if needed)
* `tmp` will contain the output images.

## Compilation

To compile the project and obtained the binary use the command `make all`. Note that it might be needed to change `ARCH_FLAGS` depending on the GPU architecture on which program is ran. It will generate multiple binary :
1. `cudart` : all optimisation except removal of virtual function
2. `cudart_sd` : all optimisation including static dispatching
3. `cudart_ref` : reference implementation (the one from Nvidia blog)
4. `cudart_ref_parallel` : the reference implementation, but only modification is the parallelisation of world creation. This is used for the validation of our version.

Each binary has two arguments : the first is the number of objects and the second one is the number of sample done by the ray tracer. For example `bin/cudart 500 20` to generate an image with roughly 500 objects and 20 samples done.

## Checking output

To check the output of the program, look at the produced image in the tmp file. `cudart` and `cudart_ref` will produce `tmp/image.ppm`. And `cudart_ref_parallel` and `cudart_ref` will respectively generate `tmp/ref_image_parallel.ppm` and `tmp/ref_image.ppm`.

For a simple test, `make test` can be used. It will compile everything needed and then launch the parallel reference, then the 2 different optimised version. MSE and SSIM metrics for image comparison are assessed, as well as timing of the different part of the program.

## Other makefile command

* `make profile_basic` and `make profile_metrics` can be used to perform basic `nvprof` profiling for the implementation containing all optimisations.
* `make clean` can be used to clean the project directory.

## Basic profiling results

This section contains basic profiling results obtained with `nvprof`.

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

### Profiling BVH implementation without world creation

BHV implementation (clean up is in parallel, but world creation is not parallel):
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

### Profiling BVH + parallel generation and clean-up

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

### Profiling BVH + parallel generation and clean-up + removal of virtual function

```
Rendering a 1200x800 image with 10 samples per pixel in 8x8 blocks.
==46379== NVPROF is profiling process 46379, command: ./bin/cudart_sd 8000
         0.198627 sec for fb_alloc
         0.008718 sec for scene_gen
         0.038886 sec for bvh_build
         0.001624 sec for render_init
         0.209414 sec for render
took 0.457372 seconds with 7925 objects.
         0.169904 sec for image_save
         0.037815 sec for cleanup
==46379== Profiling application: ./bin/cudart_sd 8000
==46379== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   96.95%  208.23ms         1  208.23ms  208.23ms  208.23ms  render(vec3_8bit*, int, int, int, camera**, bvh_flat_world**, curandStateXORWOW*)
                    1.44%  3.0940ms         1  3.0940ms  3.0940ms  3.0940ms  generate_scene_data(sphere_opt**, int)
                    0.86%  1.8438ms         1  1.8438ms  1.8438ms  1.8438ms  free_world(sphere_opt**, int, bvh_flat_world**, camera**)
                    0.72%  1.5449ms         1  1.5449ms  1.5449ms  1.5449ms  render_init(BVHNodeData const *, int, sphere_opt**, bvh_flat_world**, camera**, int, int, curandStateXORWOW*)
                    0.02%  34.369us         2  17.184us  3.8400us  30.529us  [CUDA memcpy HtoD]
                    0.01%  24.609us         1  24.609us  24.609us  24.609us  [CUDA memcpy DtoH]
                    0.00%  7.6480us         1  7.6480us  7.6480us  7.6480us  compute_bounding_boxes(sphere_opt**, int, aabb*)
                    0.00%  3.4880us         1  3.4880us  3.4880us  3.4880us  reorder_hitables(sphere_opt**, sphere_opt**, int*, int)
      API calls:   47.27%  240.60ms         1  240.60ms  240.60ms  240.60ms  cudaMallocManaged
                   42.19%  214.79ms         6  35.798ms  10.768us  208.25ms  cudaDeviceSynchronize
                    8.78%  44.706ms         1  44.706ms  44.706ms  44.706ms  cudaDeviceReset
                    1.21%  6.1507ms         6  1.0251ms  7.0820us  5.9898ms  cudaLaunchKernel
                    0.31%  1.5981ms         9  177.57us  2.8800us  1.3130ms  cudaFree
                    0.11%  574.65us         8  71.830us  3.7510us  300.70us  cudaMalloc
                    0.06%  321.50us       114  2.8200us     343ns  118.50us  cuDeviceGetAttribute
                    0.04%  226.91us         3  75.636us  42.649us  137.65us  cudaMemcpy
                    0.01%  28.675us         1  28.675us  28.675us  28.675us  cuDeviceGetName
                    0.00%  20.424us         1  20.424us  20.424us  20.424us  cuDeviceGetPCIBusId
                    0.00%  7.2980us         1  7.2980us  7.2980us  7.2980us  cuDeviceTotalMem
                    0.00%  4.9270us         3  1.6420us     502ns  3.6470us  cuDeviceGetCount
                    0.00%  3.9700us         6     661ns     152ns  1.8140us  cudaGetLastError
                    0.00%  2.2820us         2  1.1410us     425ns  1.8570us  cuDeviceGet
                    0.00%     926ns         1     926ns     926ns     926ns  cuModuleGetLoadingMode
                    0.00%     638ns         1     638ns     638ns     638ns  cuDeviceGetUuid

==46379== Unified Memory profiling result:
Device "NVIDIA GeForce GTX 1080 Ti (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      25  112.64KB  4.0000KB  0.9961MB  2.750000MB  258.0310us  Device To Host
      12         -         -         -           -  1.561779ms  Gpu page fault groups
Total CPU Page faults: 12
```