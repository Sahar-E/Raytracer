//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include "Vec3.h"
#include "Ray.hpp"
#include "HitResult.h"
#include "Material.h"

class Sphere {

public:
    Sphere(const Point3 &center, double radius, const Material& mat) : _center(center), _radius(radius), _material(mat) {}

    bool hit(const Ray &ray, double tStart, double tEnd, HitResult &hitRes) const;

    [[nodiscard]] const Material &getMaterial() const {
        return _material;
    }

private:
    Point3 _center;
    double _radius;
    Material _material;
};



