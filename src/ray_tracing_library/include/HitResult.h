//
// Created by Sahar on 11/06/2022.
//

#pragma once

#include <memory>
#include "Vec3.h"
#include "Ray.hpp"
#include "Material.h"

class Material;

struct HitResult {
    Ray hittingRay{};
    double tOfHittingRay{};
    Point3 hitPoint{};
    Vec3 normal{};
    std::shared_ptr<Material> material;
};