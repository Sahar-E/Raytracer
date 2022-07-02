//
// Created by Sahar on 11/06/2022.
//

#include "Vec3.cuh"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "my_math_cuda.cuh"
#include "my_math.h"

__host__ __device__ Vec3 &Vec3::operator+=(const Vec3 &v) {
    _x[0] += v._x[0];
    _x[1] += v._x[1];
    _x[2] += v._x[2];
    return *this;
}

__host__ __device__ Vec3 &Vec3::operator*=(const double t) {
    _x[0] *= t;
    _x[1] *= t;
    _x[2] *= t;
    return *this;
}

__host__ __device__ Vec3 &Vec3::operator/=(const double t) {
    return *this *= 1 / t;
}

__host__ __device__ double Vec3::length() const {
    return sqrt(length_squared());
}

__host__ __device__ double Vec3::length_squared() const {
    return _x[0] * _x[0] + _x[1] * _x[1] + _x[2] * _x[2];
}

__host__ __device__ Vec3 Vec3::operator-() const {
    return {-_x[0], -_x[1], -_x[2]};
}


__host__ __device__ Vec3 operator+(const Vec3 &u, const Vec3 &v) {
    return {u.x() + v.x(), u.y() + v.y(), u.z() + v.z()};
}

__host__ __device__ Vec3 operator-(const Vec3 &u, const Vec3 &v) {
    return {u.x() - v.x(), u.y() - v.y(), u.z() - v.z()};
}

__host__ __device__ Vec3 operator*(const Vec3 &u, const Vec3 &v) {
    return {u.x() * v.x(), u.y() * v.y(), u.z() * v.z()};
}

__host__ __device__ Vec3 operator*(double t, const Vec3 &v) {
    return {t * v.x(), t * v.y(), t * v.z()};
}

__host__ __device__ Vec3 operator*(const Vec3 &v, double t) {
    return t * v;
}

__host__ __device__ Vec3 operator/(Vec3 v, double t) {
    return (1.0 / t) * v;
}

__host__ __device__ double dot(const Vec3 &u, const Vec3 &v) {
    return u.x() * v.x()
           + u.y() * v.y()
           + u.z() * v.z();
}

__host__ __device__ Vec3 cross(const Vec3 &u, const Vec3 &v) {
    return {u.y() * v.z() - u.z() * v.y(),
            u.z() * v.x() - u.x() * v.z(),
            u.x() * v.y() - u.y() * v.x()};
}

__host__ __device__ Vec3 normalize(Vec3 v) {
    return v / v.length();
}

__host__ __device__ Vec3 reflect(const Vec3 &v, const Vec3 &n) {
    return v - 2 * dot(v, n) * n;
}

__host__ __device__ Vec3 refract(const Vec3 &rayDirNormalized, const Vec3 &n, double refractionIdxRatio) {
    double cosTheta = fmin(dot(-rayDirNormalized, n), 1.0);
    Vec3 r_out_perp = refractionIdxRatio * (rayDirNormalized + cosTheta * n);
    Vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__host__ Vec3 randomVec0to1(int &randState) {
    return {randomDouble(randState),
            randomDouble(randState),
            randomDouble(randState)};
}

__device__ Vec3 randomVec0to1(curandState *randState) {
    return {randomFloatCuda(randState),
            randomFloatCuda(randState),
            randomFloatCuda(randState)};
}

__host__ Vec3 randomUnitVec(int &randState) {
    return normalize({2 * randomDouble(randState) - 1, 2 * randomDouble(randState) - 1, 2 * randomDouble(randState) - 1});
}

__device__ Vec3 randomUnitVec(curandState *randState) {
    return normalize({2 * randomFloatCuda(randState) - 1, 2 * randomFloatCuda(randState) - 1, 2 * randomFloatCuda(randState) - 1});
}

__host__ Vec3 randomVecInUnitDisk(int &randState) {
    while (true) {
        auto p = randomUnitVec(randState);
        if (1 <= p.length_squared()) {
            continue;
        }
        return p;
    }
}

__device__ Vec3 randomVecInUnitDisk(curandState *randState) {
    while (true) {
        auto p = randomUnitVec(randState);
        if (1 <= p.length_squared()) {
            continue;
        }
        return p;
    }
}
//
//__host__ __device__ Vec3 randomInUnitSphere(int &randState) {
//    while (true) {
//        auto p = randomVec(randState, 0, randState);
//        if (1 <= p.length_squared()) {
//            return p;
//        }
//    }
//}
//
//__host__ __device__ Vec3 randomInHemisphere(const Vec3 &normal) {
//    Vec3 in_unit_sphere = randomInUnitSphere(<#initializer#>);
//    if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
//        return in_unit_sphere;
//    else
//        return -in_unit_sphere;
//}

__host__ __device__ bool isZeroVec(const Vec3 &v) {
    return fcmp(v.x(), 0) && fcmp(v.y(), 0) && fcmp(v.z(), 0);
}

__host__ __device__
Vec3 alphaBlending(Vec3 v1, Vec3 v2, double alpha) {
    return v1 * (1 - alpha) + v2 * alpha;
}

__host__ __device__
Color gammaCorrection(const Color &color) {
    return {std::sqrt(color.x()), std::sqrt(color.y()), std::sqrt(color.z())};
}

__host__ __device__
Vec3 clamp(const Vec3 &toClamp, double low, double high) {
    return {clamp(toClamp.x(), low, high),
            clamp(toClamp.y(), low, high),
            clamp(toClamp.z(), low, high)};
}
