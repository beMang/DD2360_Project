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