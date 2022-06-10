//
// Created by Sahar on 10/06/2022.
//

#include "../include/Sphere.h"

Sphere::Sphere(const Point3 &center, double radius) : _center(center), _radius(radius) {}

bool Sphere::hit(const Ray &ray, double t_start, double t_end, Color &color, double & tHit) const {
    Vec3 oc = ray.origin() - _center;
    auto a = ray.direction().length_squared();
    auto half_b = dot(oc, ray.direction());
    auto c = oc.length_squared() - _radius*_radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) {
        return false;
    }
    auto sqrtD = std::sqrt(discriminant);
    auto root = (-half_b - sqrtD) / a;      // Closer root
    if (root < t_start || root > t_end) {
        root = (-half_b + sqrtD) / a;       // Farther root
        if (root < t_start || root > t_end) {
            return false;
        }
    }
    color = {0.5, 1, 0.5};
    tHit = root;
    return true;
}
