//
// Created by Sahar on 10/06/2022.
//

#pragma once

#include "cuda_runtime_api.h"
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include "World.cuh"
#include "Camera.cuh"

class RayTracerRenderer {

public:

    RayTracerRenderer(int imageWidth, int imageHeight, const World &world, std::shared_ptr<Camera> camera, int rayBounces);
    virtual ~RayTracerRenderer();

    RayTracerRenderer() = delete;
    RayTracerRenderer(const RayTracerRenderer &other) = delete;
    RayTracerRenderer &operator=(const RayTracerRenderer &other) = delete;

    void render();

    void clearPixels();

    void syncPixelsOutAsChars();

    std::shared_ptr<unsigned char[]> getPixelsOutAsChars() const;
    [[nodiscard]] int getImgW() const;
    [[nodiscard]] int getImgH() const;

    [[nodiscard]] int getNRayBounces() const;
    void setNRayBounces(int nRayBounces);

private:
    int _imgW;
    int _imgH;
    int _sharedMemForSpheres;
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

    static World **allocateWorldInDeviceMemory(const Sphere *ptrSpheres, size_t nSpheres);
    static void freeWorldFromDeviceAndItsPtr(World **d_world);
    void initRandStates();
    void initPixelAllocations();
};



