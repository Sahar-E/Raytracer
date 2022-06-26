//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include <vector>
#include "Sphere.cuh"

static const double CLOSEST_POSSIBLE_RAY_HIT = 0.001;

//class ArrayOfObjects {
//public:
//    __host__ __device__ ArrayOfObjects(const Sphere *spheresArr, size_t numSpheres);
//    __host__ __device__ ArrayOfObjects(const ArrayOfObjects &other);
//    __host__ __device__ ArrayOfObjects &operator=(ArrayOfObjects other);
//    __host__ __device__ void swap(ArrayOfObjects &first, ArrayOfObjects &second);
//    __host__ __device__ virtual ~ArrayOfObjects();
//
//    __host__ __device__ [[nodiscard]] size_t getNSpheres() const;
//    __host__ __device__ void setNSpheres(size_t nSpheres);
//
//    __host__ __device__ [[nodiscard]] Sphere *getSpheres() const;
//    __host__ __device__ void setSpheres(Sphere *spheres);
//
//private:
//    size_t _nSpheres{};
//    Sphere *_spheres{};
//};

class World {
public:
    __host__ __device__ World(const Sphere *spheresArr, size_t numSpheres) : _spheres(spheresArr), _nSpheres(numSpheres){}

    __host__ __device__ Color rayTrace(const Ray &ray, int bounce, int &randState) const;
    __host__ __device__ static Color backgroundColor(const Ray &ray);
    __host__ __device__ bool getHitResult(const Ray &ray, HitResult &hitRes, Material &material) const;

//    __host__ __device__ [[nodiscard]] const auto &getObjects() const {
//        return _objects;
//    }
    size_t getNSpheres() const;

    const Sphere *getSpheres() const;

private:
    /**
     * Used for just holding the World resources.
     */

    size_t _nSpheres{};
    const Sphere *_spheres{};
//     Private fields:
//    ArrayOfObjects _objects;
};
