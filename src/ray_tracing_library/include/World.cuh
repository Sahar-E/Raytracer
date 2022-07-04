//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include <vector>
#include "Sphere.cuh"

#define CLOSEST_POSSIBLE_RAY_HIT 0.001


class World {
public:
    __host__ __device__ World(const Sphere *spheresArr, int numSpheres);
    __host__ __device__ World(const World &other);
    __host__ __device__ World &operator=(World other);
    __host__ __device__ virtual ~World();
    __host__ __device__ static void swap(World &first, World &second);


    __host__ __device__ static Color backgroundColor(const Ray &ray);


    __host__ __device__ int getNSpheres() const;
    __host__ __device__ Sphere *getSpheres() const;

    __host__ __device__ int getTotalSizeInMemoryForObjects() const;

    static World initWorld1();
    static World initWorld2();

private:
    int _nSpheres{};
    Sphere *_spheres{};
};
