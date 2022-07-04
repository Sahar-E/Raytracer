//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include <vector>
#include "Sphere.cuh"

#define CLOSEST_POSSIBLE_RAY_HIT 0.001


class World {
public:
    __host__ __device__ World(const Sphere *spheresArr, size_t numSpheres);
    __host__ __device__ World(const World &other);
    __host__ __device__ World &operator=(World other);
    __host__ __device__ virtual ~World();
    __host__ __device__ static void swap(World &first, World &second);


    __device__ Color static rayTrace(const Sphere *spheres, size_t n_spheres, const Ray &ray, int bounce,
                                     curandState *randState);

    __host__ __device__ static Color backgroundColor(const Ray &ray);

    __host__ __device__ bool static getHitResult(Sphere *spheres, size_t n_spheres, const Ray &ray, HitResult &hitRes,
                                                 Material &material);


    __host__ __device__ size_t getNSpheres() const;
    __host__ __device__ Sphere *getSpheres() const;

    __host__ __device__ int getTotalSizeInSharedMemory() const;

    static World initWorld1();
    static World initWorld2();

private:
    size_t _nSpheres{};
    Sphere *_spheres{};
};
