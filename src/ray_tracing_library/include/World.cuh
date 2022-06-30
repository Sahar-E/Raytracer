//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include <vector>
#include "Sphere.cuh"

#define CLOSEST_POSSIBLE_RAY_HIT 0.001

//class ArrayOfObjects {
//public:
//    __host__ __device__ ArrayOfObjects(const Sphere *spheresArr, size_t numSpheres);
//
//    __host__ __device__ ArrayOfObjects(const ArrayOfObjects &other);
//
////    __host__ __device__ ArrayOfObjects &operator=(ArrayOfObjects other);
//
////    __host__ __device__ void swap(ArrayOfObjects &first, ArrayOfObjects &second);
//
//    __host__ __device__ virtual ~ArrayOfObjects();
//
//    __host__ __device__ size_t getNSpheres() const;
//
//    __host__ __device__ void setNSpheres(size_t nSpheres);
//
//    __host__ __device__ Sphere *getSpheres() const;
//
//    __host__ __device__ void setSpheres(Sphere *spheres);
//
//private:
//    size_t _nSpheres{};
//    Sphere *_spheres{};
//};

class World {
public:
    __host__ __device__
    World(const Sphere *spheresArr, size_t numSpheres)
            : _spheres(numSpheres <= 0 && spheresArr != nullptr ? nullptr : new Sphere[numSpheres]),
              _nSpheres(numSpheres) {
        for (int i = 0; i < _nSpheres; ++i) {
            _spheres[i] = spheresArr[i];
        }
    }

    __host__ __device__ World &operator=(World other) {
        swap(*this, other);
        return *this;
    }

    __host__ __device__ virtual ~World() {
        if (_spheres != nullptr) {
            delete[] _spheres;
            _spheres = nullptr;
        }
    }

    __host__ __device__ static void swap(World &first, World &second) {
        auto temp1 = second._nSpheres;
        second._nSpheres = first._nSpheres;
        first._nSpheres = temp1;

        auto temp2 = second._spheres;
        second._spheres = first._spheres;
        first._spheres = temp2;
    }


    __host__ __device__ Color rayTrace(const Ray &ray, int bounce, int &randState) const;

    __host__ __device__ static Color backgroundColor(const Ray &ray);

    __host__ __device__ bool getHitResult(const Ray &ray, HitResult &hitRes, Material &material) const;


    size_t getNSpheres() const;

    Sphere *getSpheres() const;

private:
    /**
     * Used for just holding the World resources.
     */

    size_t _nSpheres{};
    Sphere *_spheres{};
//     Private fields:
//    ArrayOfObjects _objects;
};
