//
// Created by Sahar on 08/06/2022.
//

#pragma once

#include "Vec3.h"

class Ray {
public:
    Ray() = default;

    Ray(const Point3 &origin, const Vec3 &direction) : orig(origin), dir(direction) {}

    [[nodiscard]] Point3 origin() const { return orig; }

    [[nodiscard]] Vec3 direction() const { return dir; }

    [[nodiscard]] Point3 at(double t) const {
        return orig + dir * t;
    }
public:
    Point3 orig;
    Vec3 dir;
};


inline std::ostream &operator<<(std::ostream &out, const Ray &r) {
    return out << r.origin() << " + t * " << r.direction();
}

inline Ray getZeroRay() {
    return {{0, 0, 0},
            {0, 0, 0}};
}