//
// Created by Sahar on 08/06/2022.
//

#pragma once
#include <iostream>
#include <complex>

template<typename T = double>
class Vec3 {
public:
    Vec3() : _x{0, 0, 0}{}

    Vec3(T x, T y, T z) : _x{x, y, z} {}

    [[nodiscard]] T x() const { return _x[0]; }

    [[nodiscard]] T y() const { return _x[1]; }

    [[nodiscard]] T z() const { return _x[2]; }

    Vec3 operator-() const {
        return {-_x[0], -_x[1], -_x[2]};
    }

    T operator[](int i) const { return _x[i]; }

    T &operator[](int i) { return _x[i]; }

    Vec3 &operator+=(const Vec3 &v) {
        _x[0] += v._x[0];
        _x[1] += v._x[1];
        _x[2] += v._x[2];
        return *this;
    }

    Vec3 &operator*=(const T t) {
        _x[0] *= t;
        _x[1] *= t;
        _x[2] *= t;
        return *this;
    }

    Vec3 &operator/=(const T t) {
        return *this *= 1 / t;
    }

    [[nodiscard]] T length() const {
        return std::sqrt(length_squared());
    }

    [[nodiscard]] T length_squared() const {
        return _x[0] * _x[0] + _x[1] * _x[1] + _x[2] * _x[2];
    }

private:
    T _x[3];
};

inline std::ostream& operator<<(std::ostream &out, const Vec3<> &v) {
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

[[nodiscard]] inline Vec3<> operator+(const Vec3<> &u, const Vec3<> &v) {
    return {u.x() + v.x(), u.y() + v.y(), u.z() + v.z()};
}

[[nodiscard]] inline Vec3<> operator-(const Vec3<> &u, const Vec3<> &v) {
    return {u.x() - v.x(), u.y() - v.y(), u.z() - v.z()};
}


[[nodiscard]] inline Vec3<> operator*(const Vec3<> &u, const Vec3<> &v) {
    return {u.x() * v.x(), u.y() * v.y(), u.z() * v.z()};
}

template <typename T>
[[nodiscard]] inline Vec3<> operator*(T t, const Vec3<> &v) {
    return {t*v.x(), t*v.y(), t*v.z()};
}

template <typename T>
[[nodiscard]] inline Vec3<> operator*(const Vec3<> &v, T t) {
    return t * v;
}

template <typename T>
[[nodiscard]] inline Vec3<> operator/(Vec3<> v, T t) {
    return (1/t) * v;
}

template <typename T>
[[nodiscard]] inline T dot(const Vec3<> &u, const Vec3<> &v) {
    return u.x() * v.x()
           + u.y() * v.y()
           + u.z() * v.z();
}

[[nodiscard]] inline Vec3<> cross(const Vec3<> &u, const Vec3<> &v) {
    return {u.y() * v.z() - u.z() * v.y(),
            u.z() * v.x() - u.x() * v.z(),
            u.x() * v.y() - u.y() * v.x()};
}

[[nodiscard]] inline Vec3<> unit_vector(Vec3<> v) {
    return v / v.length();
}