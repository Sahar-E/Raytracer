//
// Created by Sahar on 08/06/2022.
//

#pragma once

#include <iostream>
#include <complex>

class Vec3 {
public:
    Vec3() : _x{0, 0, 0} {}

    Vec3(double x, double y, double z) : _x{x, y, z} {}

    explicit Vec3(const double arr[3]) : Vec3(arr[0], arr[1], arr[2]) {}

    [[nodiscard]] double x() const { return _x[0]; }

    [[nodiscard]] double y() const { return _x[1]; }

    [[nodiscard]] double z() const { return _x[2]; }

    Vec3 operator-() const {
        return {-_x[0], -_x[1], -_x[2]};
    }

    double operator[](int i) const { return _x[i]; }

    double &operator[](int i) { return _x[i]; }

    Vec3 &operator+=(const Vec3 &v) {
        _x[0] += v._x[0];
        _x[1] += v._x[1];
        _x[2] += v._x[2];
        return *this;
    }

    Vec3 &operator*=(const double t) {
        _x[0] *= t;
        _x[1] *= t;
        _x[2] *= t;
        return *this;
    }

    Vec3 &operator/=(const double t) {
        return *this *= 1 / t;
    }

    [[nodiscard]] double length() const {
        return std::sqrt(length_squared());
    }

    [[nodiscard]] double length_squared() const {
        return _x[0] * _x[0] + _x[1] * _x[1] + _x[2] * _x[2];
    }

private:
    double _x[3];
};


// Using Vec3 with the following aliases:
using Point3 = Vec3;
using Color = Vec3;


inline std::ostream &operator<<(std::ostream &out, const Vec3 &v) {
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

[[nodiscard]] inline Vec3 operator+(const Vec3 &u, const Vec3 &v) {
    return {u.x() + v.x(), u.y() + v.y(), u.z() + v.z()};
}

[[nodiscard]] inline Vec3 operator-(const Vec3 &u, const Vec3 &v) {
    return {u.x() - v.x(), u.y() - v.y(), u.z() - v.z()};
}


[[nodiscard]] inline Vec3 operator*(const Vec3 &u, const Vec3 &v) {
    return {u.x() * v.x(), u.y() * v.y(), u.z() * v.z()};
}

[[nodiscard]] inline Vec3 operator*(double t, const Vec3 &v) {
    return {t * v.x(), t * v.y(), t * v.z()};
}

[[nodiscard]] inline Vec3 operator*(const Vec3 &v, double t) {
    return t * v;
}

[[nodiscard]] inline Vec3 operator/(Vec3 v, double t) {
    return (1.0 / t) * v;
}

[[nodiscard]] inline double dot(const Vec3 &u, const Vec3 &v) {
    return u.x() * v.x()
           + u.y() * v.y()
           + u.z() * v.z();
}

[[nodiscard]] inline Vec3 cross(const Vec3 &u, const Vec3 &v) {
    return {u.y() * v.z() - u.z() * v.y(),
            u.z() * v.x() - u.x() * v.z(),
            u.x() * v.y() - u.y() * v.x()};
}

[[nodiscard]] inline Vec3 unit_vector(Vec3 v) {
    return v / v.length();
}