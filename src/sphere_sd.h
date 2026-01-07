#ifndef SPHERE_SD_H
#define SPHERE_SD_H

#include "hitable_sd.h"

/**
 * Optimized Sphere representation.
 * Implements intersection logic for ray-sphere tests using the optimized hit record.
 */
class sphere_opt {
    public:
        __device__ sphere_opt() {}
        __device__ sphere_opt(vec3 cen, float r, material_opt *m) : center(cen), radius(r), mat(m) {};

        __device__ virtual aabb bounding_box() const;

        /**
         * Standard quadratic formula for ray-sphere intersection.
         * Updates the hit_record_opt with the closest intersection within [t_min, t_max].
         */
        __device__ bool hit(const ray& r, float t_min, float t_max, hit_record_opt& rec) const {
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
                    rec.mat_ptr = mat;
                    return true;
                }
                temp = (-b + sqrt(discriminant)) / a;
                if (temp < t_max && temp > t_min) {
                    rec.t = temp;
                    rec.p = r.point_at_parameter(rec.t);
                    rec.normal = (rec.p - center) / radius;
                    rec.mat_ptr = mat;
                    return true;
                }
            }
            return false;
        }

        vec3 center;
        float radius;
        material_opt *mat;
};

/**
 * Calculates the Axis-Aligned Bounding Box (AABB) for the sphere.
 */
__device__ aabb sphere_opt::bounding_box() const {
    vec3 r(radius, radius, radius);
    return aabb(center - r, center + r);
}

#endif