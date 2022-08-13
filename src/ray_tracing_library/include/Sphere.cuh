//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include "HitResult.h"
#include "Material.cuh"

/**
 * Represent a sphere in the world.
 */
class Sphere {
public:
    __host__ __device__ Sphere() = default;

    /**
     * Return a new instance of the sphere.
     * @param center    the center of the sphere.
     * @param radius    the radius of the sphere.
     * @param mat       the material of the sphere.
     */
    __host__ __device__ Sphere(const Point3 &center, float radius, const Material& mat)
            : _center(center), _radius(radius), _material(mat) {}

    /**
     * Get hit result for a ray that hit the sphere.
     * @param ray       the ray that had hit the sphere.
     * @param rootRes   the root res of the ray hit.
     * @param hitRes    where to store the hit result.
     */
    __host__ __device__ void getHitResult(const Ray &ray, float rootRes, HitResult &hitRes) const;

    /**
     * Check if the ray hit the sphere.
     * @param ray       The ray that is casted in the scene.
     * @param tStart    Will check for intersection between the ray and the sphere starting from tStart along the ray.
     * @param tEnd      Will check for intersection between the ray and the sphere ending at tEnd along the ray.
     * @param rootRes   Will write the root res of the ray hit here.
     * @return  true if the ray hit the sphere, false otherwise.
     */
    __host__ __device__ bool isHit(const Ray &ray, float tStart, float tEnd, float &rootRes) const;

    /**
     * @return  the sphere material.
     */
    __host__ __device__ const Material &getMaterial() const;

private:
    Point3 _center{};
    float _radius{};
    Material _material{};
};



