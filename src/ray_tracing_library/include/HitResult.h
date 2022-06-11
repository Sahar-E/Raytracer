//
// Created by Sahar on 11/06/2022.
//

#pragma once
#include "Vec3.h"
#include "Ray.hpp"


struct HitResult {
    Ray reflectionRay{};
    Color color{};
    double tOfHittingRay{};
    Point3 hitPoint{};
    Vec3 normal{};
};