//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include "HitResult.h"
#include "Material.cuh"

class Sphere {

public:
    __host__ __device__ Sphere() = default;

    __host__ __device__ Sphere(const Point3 &center, float radius, const Material& mat)
            : _center(center), _radius(radius), _material(mat) {}

    __host__ __device__ void
    getHitResult(const Ray &ray, float rootRes, HitResult &hitRes) const;
    __host__ __device__ bool isHit(const Ray &ray, float tStart, float tEnd, float &rootRes) const;

    __host__ __device__ const Material &getMaterial() const;

private:
    Point3 _center{};
    float _radius{};
    Material _material{};
};



