//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include <Vec3.hpp>
#include <Ray.hpp>

class Sphere {

public:
    Sphere(const Point3 &center, double radius);

    bool hit(const Ray &ray, double t_start, double t_end, Color &color, double & tHit) const;

private:
    Point3 _center;
    double _radius;

};



