//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include <vector>
#include "Sphere.cuh"

#define CLOSEST_POSSIBLE_RAY_HIT 0.0005

/**
 * This class represents a scene that can be used to render ray casted images.
 */
class World {
public:

    /**
     * Construct a new world instance.
     * @param spheresArr    The array of objects spheres in the world.
     * @param numSpheres    The numberr of spheres in the world.
     */
    __host__ __device__ World(const Sphere *spheresArr, int numSpheres);
    __host__ __device__ World(const World &other);
    __host__ __device__ World &operator=(World other);
    __host__ __device__ virtual ~World();
    __host__ __device__ static void swap(World &first, World &second);

    /**
     * Background color of the world when no objects were hit.
     * @param ray   The ray that was casted to the world.
     * @return  The resulted color for that ray cast.
     */
    __host__ __device__ static Color backgroundColor(const Ray &ray);

    __host__ __device__ int getNSpheres() const;
    __host__ __device__ Sphere *getSpheres() const;

    /**
     * @return  the total memory in bytes for the world objects.
     */
    __host__ __device__ int getTotalSizeInMemoryForObjects() const;

    /**
     * @return  World scene with 8~ sphere objects.
     */
    static World initWorld1();

    /**
     * @return  World scene with 50~ sphere objects.
     */
    static World initWorld2();

private:
    int _nSpheres{};
    Sphere *_spheres{};
};
