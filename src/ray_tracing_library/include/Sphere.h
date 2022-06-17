//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include <Vec3.h>
#include <Ray.hpp>
#include "HitResult.h"

class Sphere {

public:
    Sphere(const Point3 &center, double radius, std::shared_ptr<Material> material);

    bool hit(const Ray &ray, double tStart, double tEnd, HitResult &hitRes) const;

private:
    Point3 _center;
    double _radius;
    std::shared_ptr<Material> _material;
};



