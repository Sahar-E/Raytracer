//
// Created by Sahar on 08/06/2022.
//

#pragma once

#include "my_math.hpp"
#include <iostream>
#include <complex>

class Vec3 {
public:
    Vec3() : _x{0, 0, 0} {}

    Vec3(double x, double y, double z) : _x{x, y, z} {}

    [[nodiscard]] double x() const { return _x[0]; }

    [[nodiscard]] double y() const { return _x[1]; }

    [[nodiscard]] double z() const { return _x[2]; }

    Vec3 operator-() const;

    inline double operator[](int i) const { return _x[i]; }

    inline double &operator[](int i) { return _x[i]; }

    Vec3 &operator+=(const Vec3 &v);

    Vec3 &operator*=(double t);

    Vec3 &operator/=(double t);

    [[nodiscard]] double length() const;

    [[nodiscard]] double length_squared() const;



private:
    double _x[3];
};


// Using Vec3 with the following aliases:
using Point3 = Vec3;
using Color = Vec3;


std::ostream &operator<<(std::ostream &out, const Vec3 &v);

[[nodiscard]] Vec3 operator+(const Vec3 &u, const Vec3 &v);

[[nodiscard]] Vec3 operator-(const Vec3 &u, const Vec3 &v);

[[nodiscard]] Vec3 operator*(const Vec3 &u, const Vec3 &v);

[[nodiscard]] Vec3 operator*(double t, const Vec3 &v);

[[nodiscard]] Vec3 operator*(const Vec3 &v, double t);

[[nodiscard]] Vec3 operator/(Vec3 v, double t);

[[nodiscard]] double dot(const Vec3 &u, const Vec3 &v);

[[nodiscard]] Vec3 cross(const Vec3 &u, const Vec3 &v);

[[nodiscard]] Vec3 unitVector(Vec3 v);

Vec3 reflect(const Vec3& v, const Vec3& n);


Vec3 randomVec();

Vec3 randomUnitVec();

Vec3 randomInUnitSphere();

Vec3 randomInHemisphere(const Vec3 &normal);

Vec3 randomVecOnTangentSphere(const Vec3 &normal, const Vec3 & hitPoint);

bool isZeroVec(const Vec3 &v);

bool isZeroVec(const Vec3 &v);