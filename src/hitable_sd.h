#ifndef HITABLE_SD_H
#define HITABLE_SD_H

#include "ray.h"
#include "aabb.h"

class material_opt; 

/**
 * Optimized hit record structure.
 * Stores information about a ray-object intersection.
 * mat_ptr points to a struct-based material to avoid virtual function overhead.
 */
struct hit_record_opt {
    float t;             // Distance along the ray
    vec3 p;              // Intersection point coordinates
    vec3 normal;         // Surface normal at intersection
    material_opt *mat_ptr; 
};

/**
 * Base class for hittable objects.
 * Note: While virtual functions are generally avoided in the "Static dispatch" 
 * architecture for materials, this interface remains for geometric abstraction.
 */
class hitable_opt {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record_opt& rec) const = 0;
};

#endif