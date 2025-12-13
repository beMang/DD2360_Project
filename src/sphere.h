#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere: public hitable  {
    public:
        __device__ sphere() {}
        __device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m)  {};
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        __device__ virtual aabb bounding_box() const;
        vec3 center;
        float radius;
        material *mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

/**
 * Compute the bounding box for the sphere
 * @return AABB that contains the sphere
 */
__device__ aabb sphere::bounding_box() const {
    vec3 r(radius, radius, radius);
    return aabb(center - r, center + r);
}

// Host-side scene description used to build a BVH on CPU then transfer to GPU.
struct SphereData {
    vec3 center;
    float radius;
    int mat_type;   // 0 lambertian, 1 metal, 2 dielectric
    vec3 albedo;
    float fuzz;
    float ref_idx;
};

// Host-side function to compute box for SphereData
inline aabb box_for_sphere(const SphereData& s) {
    vec3 r(s.radius, s.radius, s.radius);
    return aabb(s.center - r, s.center + r);
}


#endif
