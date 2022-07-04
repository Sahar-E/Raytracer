//
// Created by Sahar on 08/06/2022.
//

#pragma once

#include "Vec3.cuh"

class Ray {
public:
    __host__ __device__ Ray() = default;

    __host__ __device__ Ray(const Point3 &origin, const Vec3 &direction) : _orig(origin), _dir(direction) {}

    __host__ __device__  Point3 origin() const { return _orig; }

    __host__ __device__  Vec3 direction() const { return _dir; }

    __host__ __device__  Point3 at(float t) const {
        return _orig + _dir * t;
    }

    __host__ __device__  bool isZeroRay() const {
        return isZeroVec(_orig) && isZeroVec(_dir);
    }

public:
    Point3 _orig;
    Vec3 _dir;
};
