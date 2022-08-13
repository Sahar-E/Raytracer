//
// Created by Sahar on 10/06/2022.
//

#pragma once

#include "cuda_runtime_api.h"
#include <vector>
#include <memory>
#include "World.cuh"
#include "Camera.cuh"


/**
 * The RayTracerRenderer is responsible for rendering a ray-traced image in a given world scene.
 */
class RayTracerRenderer {
public:

    /**
     * Creates a new RayTracerRenderer.
     * @param imageWidth    The width of the image, number of pixels.
     * @param imageHeight   The height of the image, number of pixels.
     * @param world         The world scene.
     * @param camera        The camera in the world scene.
     * @param rayBounces    The number of ray bounces to perform for each ray when casted in to the world.
     */
    RayTracerRenderer(int imageWidth, int imageHeight, const World &world, std::shared_ptr<Camera> camera, int rayBounces);
    virtual ~RayTracerRenderer();

    RayTracerRenderer() = delete;
    RayTracerRenderer(const RayTracerRenderer &other) = delete;
    RayTracerRenderer &operator=(const RayTracerRenderer &other) = delete;

    /**
     * Render a view of the scene with the camera, and average the result into the last frame.
     */
    void render();

    /**
     * Clear the image buffer.
     */
    void clearPixels();

    /**
     * Called to sync the image buffer in the GPU with the read buffer accessible for the CPU with the method
     * getPixelsOutAsChars().
     */
    void syncPixelsOutAsChars();

    /**
     * Get a pointer to read the image buffer. Should syncPixelsOutAsChars to get the latest result.
     * @return  pointer to read the image buffer.
     */
    [[nodiscard]] std::shared_ptr<unsigned char[]> getPixelsOutAsChars() const;

    [[nodiscard]] int getImgW() const;
    [[nodiscard]] int getImgH() const;

    [[nodiscard]] int getNRayBounces() const;
    void setNRayBounces(int nRayBounces);

private:
    int _imgW;
    int _imgH;
    int _d_SharedMemForSpheres;
    World** _d_world;
    std::shared_ptr<Camera> _camera;
    int _nRayBounces;
    float _alreadyNPixelsGot{};
    curandState *_randStates;
    Color *_pixelsOut_cuda{};
    Color *_innerPixelsAverageAccum_cuda{};
    std::shared_ptr<unsigned char[]> _pixelsOutAsChars{};
    std::shared_ptr<Color []> _pixelsOutAsColors{};
    Ray *_rays{};

    /**
     * Init the world inside the GPU memory.
     * @param ptrSpheres    Spheres array.
     * @param nSpheres      number of spheres.
     * @return  pointer to the initialized world.
     */
    static World **allocateWorldInDeviceMemory(const Sphere *ptrSpheres, size_t nSpheres);

    /**
     * Free the world inside the GPU memory.
     * @param d_world  pointer to the initialized world.
     */
    static void freeWorldFromDeviceAndItsPtr(World **d_world);

    /**
     * Init the random states inside the GPU memory.
     */
    void initRandStates();

    /**
     * Init the Image buffers inside the GPU memory.
     */
    void initImgBuffersAllocations();
};



