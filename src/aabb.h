#ifndef AABBH
#define AABBH

#include <math.h>
#include "ray.h"

/**
 * Axis-Aligned Bounding Box (AABB) class
 * Represents a 3D box defined by minimum and maximum corner points
 * Used for delimitng volumes in space for the BVH
 */
class aabb {
public:
    __host__ __device__ aabb() {}
    __host__ __device__ aabb(const vec3& a, const vec3& b) : _min(a), _max(b) {}

    __host__ __device__ vec3 min() const { return _min; }
    __host__ __device__ vec3 max() const { return _max; }

    /**
     * Compute the surface area of the AABB
     * Surface area is the sum of the areas of all six faces
     * @return Surface area as a float
     */
    __host__ __device__ inline float surface_area() const {
        vec3 d = _max - _min;
        return 2.0f * (d.x()*d.y() + d.y()*d.z() + d.z()*d.x());
    }

    /**
     * Check if a ray intersects the AABB within the range [tmin, tmax]
     * @param r The ray to test
     * @param tmin Minimum t value
     * @param tmax Maximum t value
     * @return True if the ray hits the AABB, false otherwise
     */
    __device__ bool hit(const ray& r, float tmin, float tmax) const {
        for (int a = 0; a < 3; a++) {
            float invD = 1.0f / r.direction()[a];
            float t0 = (_min[a] - r.origin()[a]) * invD;
            float t1 = (_max[a] - r.origin()[a]) * invD;
            if (invD < 0.0f) {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin) return false;
        }
        return true;
    }

private:
    vec3 _min;
    vec3 _max;
};

//min and max helper functions
__host__ __device__ inline float ffmin(float a, float b) { return a < b ? a : b; }
__host__ __device__ inline float ffmax(float a, float b) { return a > b ? a : b; }

/**
 * Create box that contains the 2 input boxes
 * @param box0 First AABB
 * @param box1 Second AABB
 * @return AABB that surrounds both input boxes
 */
__host__ __device__ inline aabb surrounding_box(const aabb& box0, const aabb& box1) {
    vec3 small(ffmin(box0.min().x(), box1.min().x()),
               ffmin(box0.min().y(), box1.min().y()),
               ffmin(box0.min().z(), box1.min().z()));
    vec3 big(ffmax(box0.max().x(), box1.max().x()),
             ffmax(box0.max().y(), box1.max().y()),
             ffmax(box0.max().z(), box1.max().z()));
    return aabb(small, big);
}

#endif
