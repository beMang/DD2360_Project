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