//
// Created by Sahar on 11/06/2022.
//

#pragma once

#include <memory>
#include "Ray.cuh"

/**
 * Struct that holds the hit data after the collision of ray with an object.
 */
struct HitResult {
    Ray hittingRay{};
    Vec3 normal{};
    Point3 hitPoint{};
    bool isOutwardsNormal{};

    __host__ __device__
    HitResult() = default;

    __host__ __device__
    HitResult(const Ray &hittingRay, const Vec3 &normal, const Point3 &hitPoint, bool outwardsNormal)
            : hittingRay(hittingRay), normal(normal), hitPoint(hitPoint),
              isOutwardsNormal(outwardsNormal) {}
};