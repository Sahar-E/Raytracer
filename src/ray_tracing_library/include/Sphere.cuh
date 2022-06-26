//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include "HitResult.h"
#include "Material.cuh"

class Sphere {

public:
    __host__ __device__
    Sphere() = default;

    __host__ __device__
    Sphere(const Point3 &center, double radius, const Material& mat) : _center(center), _radius(radius), _material(mat) {}

    __host__ __device__
    bool hit(const Ray &ray, double tStart, double tEnd, HitResult &hitRes) const;

    __host__ __device__
    const Material &getMaterial() const {
        return _material;
    }

private:
    Point3 _center{};
    double _radius{};
    Material _material{};
};



