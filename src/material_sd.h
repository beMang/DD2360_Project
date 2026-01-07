#ifndef MATERIAL_SD_H
#define MATERIAL_SD_H

struct hit_record_opt;

#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "hitable_sd.h"

/**
 * Device utility functions for optical calculations.
 */

__device__ inline float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ inline bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    return false;
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n) {
     return v - 2.0f * dot(v, n) * n;
}

__device__ inline vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

/**
 * Optimized Material structure using Static Dispatch.
 * This approach eliminates "Virtual Hell" by replacing polymorphism with a type tag 
 * and a switch statement, significantly reducing branch divergence and vtable lookups.
 */
enum MaterialType { LAMBERTIAN, METAL, DIELECTRIC };

struct material_opt {
    MaterialType type;
    vec3 albedo;
    float fuzz;
    float ref_idx;

    __device__ material_opt(MaterialType t, vec3 a = vec3(0,0,0), float f = 0.0f, float ri = 1.0f)
        : type(t), albedo(a), fuzz(f), ref_idx(ri) {}

    /**
     * Determines how a ray interacts with a surface based on material properties.
     */
    __device__ bool scatter(const ray& r_in, const hit_record_opt& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const {
        switch(type) {
            case LAMBERTIAN: {
                vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
                scattered = ray(rec.p, target - rec.p);
                attenuation = albedo;
                return true;
            }
            case METAL: {
                vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
                scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
                attenuation = albedo;
                return (dot(scattered.direction(), rec.normal) > 0);
            }
            case DIELECTRIC: {
                vec3 outward_normal;
                vec3 reflected = reflect(r_in.direction(), rec.normal);
                float ni_over_nt;
                attenuation = vec3(1.0, 1.0, 1.0);
                vec3 refracted;
                float reflect_prob;
                float cosine;
                if (dot(r_in.direction(), rec.normal) > 0) {
                    outward_normal = -rec.normal;
                    ni_over_nt = ref_idx;
                    cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
                    cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
                } else {
                    outward_normal = rec.normal;
                    ni_over_nt = 1.0f / ref_idx;
                    cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
                }
                if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
                    reflect_prob = schlick(cosine, ref_idx);
                else
                    reflect_prob = 1.0f;
                if (curand_uniform(local_rand_state) < reflect_prob)
                    scattered = ray(rec.p, reflected);
                else
                    scattered = ray(rec.p, refracted);
                return true;
            }
        }
        return false;
    }
};

#endif