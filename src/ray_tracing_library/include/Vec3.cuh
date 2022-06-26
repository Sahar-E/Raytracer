//
// Created by Sahar on 08/06/2022.
//

#pragma once

#include "my_math.cuh"

class Vec3 {
public:
    __host__ __device__ Vec3() = default;

    __host__ __device__ Vec3(double x, double y, double z) : _x{x, y, z} {}

    __host__ __device__ double x() const { return _x[0]; }

    __host__ __device__ double y() const { return _x[1]; }

    __host__ __device__ double z() const { return _x[2]; }

    __host__ __device__ Vec3 operator-() const;

    __host__ __device__ inline double operator[](int i) const { return _x[i]; }

    __host__ __device__ inline double &operator[](int i) { return _x[i]; }

    __host__ __device__ Vec3 &operator+=(const Vec3 &v);

    __host__ __device__ Vec3 &operator*=(double t);

    __host__ __device__ Vec3 &operator/=(double t);

    __host__ __device__ double length() const;

    __host__ __device__ double length_squared() const;


private:
    double _x[3];
};


// Using Vec3 with the following aliases:
using Point3 = Vec3;
using Color = Vec3;


__host__ __device__ Vec3 operator+(const Vec3 &u, const Vec3 &v);

__host__ __device__ Vec3 operator-(const Vec3 &u, const Vec3 &v);

__host__ __device__ Vec3 operator*(const Vec3 &u, const Vec3 &v);

__host__ __device__ Vec3 operator*(double t, const Vec3 &v);

__host__ __device__ Vec3 operator*(const Vec3 &v, double t);

__host__ __device__ Vec3 operator/(Vec3 v, double t);

__host__ __device__ double dot(const Vec3 &u, const Vec3 &v);

__host__ __device__ Vec3 cross(const Vec3 &u, const Vec3 &v);

__host__ __device__ Vec3 normalize(Vec3 v);

__host__ __device__ Vec3 reflect(const Vec3 &v, const Vec3 &n);

__host__ __device__ Vec3 refract(const Vec3 &rayDirNormalized, const Vec3 &n, double refractionIdxRatio);

__host__ __device__ Vec3 randomVec(int &randState);

__host__ __device__ Vec3 randomVec(double from, double to, int &randState);

__host__ __device__ Vec3 randomUnitVec(int &randState);
//
__host__ __device__ Vec3 randomVecInUnitDisk(int &randState);

__host__ __device__ bool isZeroVec(const Vec3 &v);


/**
 * Performs alpha blending between 2 colors.
 *
 * @param v1        First color.
 * @param v2        Second color.
 * @param alpha     Ratio of colors. (e.g. 1 will be only c1)
 * @return  new color.
 */
__host__ __device__ Vec3 alphaBlending(Vec3 v1, Vec3 v2, double alpha);

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
__host__ __device__ Vec3 clamp(const Vec3 &toClamp, double low, double high);