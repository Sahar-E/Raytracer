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

__host__ __device__ Vec3 &Vec3::operator*=(const float t) {
    _x[0] *= t;
    _x[1] *= t;
    _x[2] *= t;
    return *this;
}

__host__ __device__ Vec3 &Vec3::operator/=(const float t) {
    return *this *= 1 / t;
}

__host__ __device__ float Vec3::length() const {
    return sqrtf(length_squared());
}

__host__ __device__ float Vec3::length_squared() const {
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

__host__ __device__ Vec3 operator*(float t, const Vec3 &v) {
    return {t * v.x(), t * v.y(), t * v.z()};
}

__host__ __device__ Vec3 operator*(const Vec3 &v, float t) {
    return t * v;
}

__host__ __device__ Vec3 operator/(Vec3 v, float t) {
    return (1.0f / t) * v;
}

__host__ __device__ float dot(const Vec3 &u, const Vec3 &v) {
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

__host__ __device__ Vec3 reflect(const Vec3 &vec, const Vec3 &normal) {
    return vec - 2.0f * dot(vec, normal) * normal;
}

__host__ __device__ Vec3 refract(const Vec3 &rayDirNormalized, const Vec3 &normal, float refractionIdxRatio) {
    float cosTheta = fminf(dot(-rayDirNormalized, normal), 1.0f);
    Vec3 r_out_perp = refractionIdxRatio * (rayDirNormalized + cosTheta * normal);
    Vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * normal;
    return r_out_perp + r_out_parallel;
}

__host__ Vec3 randomVec0to1(int &randState) {
    return {randomFloat(),
            randomFloat(),
            randomFloat()};
}

__device__ Vec3 randomVec0to1(curandState *randState) {
    return {randomFloatCuda(randState),
            randomFloatCuda(randState),
            randomFloatCuda(randState)};
}

__host__ Vec3 randomUnitVec(int &randState) {
    return normalize({2.0f * randomFloat() - 1.0f, 2.0f * randomFloat() - 1.0f, 2.0f * randomFloat() - 1.0f});
}

__device__ Vec3 randomUnitVec(curandState *randState) {
    return normalize({2.0f * randomFloatCuda(randState) - 1.0f, 2.0f * randomFloatCuda(randState) - 1.0f, 2.0f * randomFloatCuda(randState) - 1.0f});
}

__host__ Vec3 randomVecInUnitDisk(int &randState) {
    while (true) {
        auto p = randomUnitVec(randState);
        if (1.0f <= p.length_squared()) {
            continue;
        }
        return p;
    }
}

__device__ Vec3 randomVecInUnitDisk(curandState *randState) {
    while (true) {
        auto p = randomUnitVec(randState);
        if (1.0f <= p.length_squared()) {
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
    return fcmp(v.x(), 0.0f) && fcmp(v.y(), 0.0f) && fcmp(v.z(), 0.0f);
}

__host__ __device__
Vec3 alphaBlending(Vec3 v1, Vec3 v2, float alpha) {
    return v1 * (1 - alpha) + v2 * alpha;
}

__host__ __device__
Color gammaCorrection(const Color &color) {
    return {sqrtf(color.x()), sqrtf(color.y()), sqrtf(color.z())};
}

__host__ __device__
Vec3 clamp(const Vec3 &toClamp, float low, float high) {
    return {clamp(toClamp.x(), low, high),
            clamp(toClamp.y(), low, high),
            clamp(toClamp.z(), low, high)};
}
