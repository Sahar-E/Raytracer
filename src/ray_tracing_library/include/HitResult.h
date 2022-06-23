//
// Created by Sahar on 11/06/2022.
//

#pragma once

#include <memory>
#include "Vec3.h"
#include "Ray.hpp"

struct HitResult {
    Ray hittingRay{};
    Vec3 normal{};
    Point3 hitPoint{};
    double tOfHittingRay{};
    bool isOutwardsNormal{};

    HitResult() = default;

    HitResult(const Ray &hittingRay,
              const Vec3 &normal,
              const Point3 &hitPoint,
              double tOfHittingRay,
              bool outwardsNormal) : hittingRay(hittingRay), normal(normal), hitPoint(hitPoint),
                                     tOfHittingRay(tOfHittingRay), isOutwardsNormal(outwardsNormal) {}
};