//
// Created by Sahar on 08/06/2022.
//

#pragma once

#include "Vec3.h"

class Ray {
public:
    Ray() = default;

    Ray(const Point3 &origin, const Vec3 &direction) : _orig(origin), _dir(direction) {}

    [[nodiscard]] Point3 origin() const { return _orig; }

    [[nodiscard]] Vec3 direction() const { return _dir; }

    [[nodiscard]] Point3 at(double t) const {
        return _orig + _dir * t;
    }

    [[nodiscard]] bool isZeroRay() const {
        return isZeroVec(_orig) && isZeroVec(_dir);
    }

public:
    Point3 _orig;
    Vec3 _dir;
};


inline std::ostream &operator<<(std::ostream &out, const Ray &r) {
    return out << r.origin() << " + t * " << r.direction();
}

inline Ray getZeroRay() {
    return {{0, 0, 0},
            {0, 0, 0}};
}