//
// Created by Sahar on 11/06/2022.
//

#include <my_math.hpp>
#include "Vec3.h"


Vec3 &Vec3::operator+=(const Vec3 &v) {
    _x[0] += v._x[0];
    _x[1] += v._x[1];
    _x[2] += v._x[2];
    return *this;
}

Vec3 &Vec3::operator*=(const double t) {
    _x[0] *= t;
    _x[1] *= t;
    _x[2] *= t;
    return *this;
}

Vec3 &Vec3::operator/=(const double t) {
    return *this *= 1 / t;
}

double Vec3::length() const {
    return std::sqrt(length_squared());
}

double Vec3::length_squared() const {
    return _x[0] * _x[0] + _x[1] * _x[1] + _x[2] * _x[2];
}

Vec3 Vec3::operator-() const {
    return {-_x[0], -_x[1], -_x[2]};
}


std::ostream &operator<<(std::ostream &out, const Vec3 &v) {
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

Vec3 operator+(const Vec3 &u, const Vec3 &v) {
    return {u.x() + v.x(), u.y() + v.y(), u.z() + v.z()};
}

Vec3 operator-(const Vec3 &u, const Vec3 &v) {
    return {u.x() - v.x(), u.y() - v.y(), u.z() - v.z()};
}

Vec3 operator*(const Vec3 &u, const Vec3 &v) {
    return {u.x() * v.x(), u.y() * v.y(), u.z() * v.z()};
}

Vec3 operator*(double t, const Vec3 &v) {
    return {t * v.x(), t * v.y(), t * v.z()};
}

Vec3 operator*(const Vec3 &v, double t) {
    return t * v;
}

Vec3 operator/(Vec3 v, double t) {
    return (1.0 / t) * v;
}

double dot(const Vec3 &u, const Vec3 &v) {
    return u.x() * v.x()
           + u.y() * v.y()
           + u.z() * v.z();
}

Vec3 cross(const Vec3 &u, const Vec3 &v) {
    return {u.y() * v.z() - u.z() * v.y(),
            u.z() * v.x() - u.x() * v.z(),
            u.x() * v.y() - u.y() * v.x()};
}

Vec3 unitVector(Vec3 v) {
    return v / v.length();
}

Vec3 reflect(const Vec3 &v, const Vec3 &n) {
    return v - 2 * dot(v, n) * n;
}

Vec3 randomVec() {
    return {2 * randomDouble() - 1, 2 * randomDouble() - 1, 2 * randomDouble() - 1};
}

Vec3 randomUnitVec() {
    return unitVector({2 * randomDouble() - 1, 2 * randomDouble() - 1, 2 * randomDouble() - 1});
}

Vec3 randomVecOnTangentSphere(const Vec3 &normal, const Vec3 &hitPoint) {
    return normal + hitPoint + randomUnitVec();
}

Vec3 randomInUnitSphere() {
    while (true) {
        auto p = randomVec();
        if (1 <= p.length_squared()) {
            return p;
        }
    }
}

Vec3 randomInHemisphere(const Vec3 &normal) {
    Vec3 in_unit_sphere = randomInUnitSphere();
    if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}
