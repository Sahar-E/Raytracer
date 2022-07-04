//
// Created by Sahar on 08/06/2022.
//

#pragma once

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "my_math_cuda.cuh"

class Vec3 {
public:
    __host__ __device__ Vec3() = default;
    __host__ __device__ Vec3(float x, float y, float z) : _x{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)} {}

    __host__ __device__ float x() const { return _x[0]; }
    __host__ __device__ float y() const { return _x[1]; }
    __host__ __device__ float z() const { return _x[2]; }

    __host__ __device__ Vec3 operator-() const;

    __host__ __device__ inline float operator[](int i) const { return _x[i]; }
    __host__ __device__ inline float &operator[](int i) { return reinterpret_cast<float &>(_x[i]); }

    __host__ __device__ Vec3 &operator+=(const Vec3 &v);
    __host__ __device__ Vec3 &operator*=(float t);
    __host__ __device__ Vec3 &operator/=(float t);

    __host__ __device__ float length() const;
    __host__ __device__ float length_squared() const;


private:
    float _x[3];
};


// Using Vec3 with the following aliases:
using Point3 = Vec3;
using Color = Vec3;


__host__ __device__ Vec3 operator+(const Vec3 &u, const Vec3 &v);
__host__ __device__ Vec3 operator-(const Vec3 &u, const Vec3 &v);
__host__ __device__ Vec3 operator*(const Vec3 &u, const Vec3 &v);
__host__ __device__ Vec3 operator*(float t, const Vec3 &v);
__host__ __device__ Vec3 operator*(const Vec3 &v, float t);
__host__ __device__ Vec3 operator/(Vec3 v, float t);

__host__ __device__ float dot(const Vec3 &u, const Vec3 &v);
__host__ __device__ Vec3 cross(const Vec3 &u, const Vec3 &v);
__host__ __device__ Vec3 normalize(Vec3 v);

__host__ __device__ Vec3 reflect(const Vec3 &v, const Vec3 &n);
__host__ __device__ Vec3 refract(const Vec3 &rayDirNormalized, const Vec3 &n, float refractionIdxRatio);

__host__ Vec3 randomVec0to1(int &randState);
__device__ Vec3 randomVec0to1(curandState *randState);
__host__ Vec3 randomUnitVec(int &randState);
__device__ Vec3 randomUnitVec(curandState *randState);
__host__ Vec3 randomVecInUnitDisk(int &randState);
__device__ Vec3 randomVecInUnitDisk(curandState *randState);

__host__ __device__ bool isZeroVec(const Vec3 &v);


/**
 * Performs alpha blending between 2 colors.
 *
 * @param v1        First color.
 * @param v2        Second color.
 * @param alpha     Ratio of colors. (e.g. 1 will be only c1)
 * @return  new color.
 */
__host__ __device__ Vec3 alphaBlending(Vec3 v1, Vec3 v2, float alpha);

/**
 * Do gamma correction (of 2.0) to the given color.
 * @param color The color to do gamma correction to.
 * @return  Corrected color.
 */
__host__ __device__ Color gammaCorrection(const Color &color);

/**
 * Clamp the vector between 2 values.
 * @param toClamp   Number to clamp.
 * @param low       Low bound.
 * @param high      High bound.
 * @return  Clamp result.
 */
__host__ __device__ Vec3 clamp(const Vec3 &toClamp, float low, float high);