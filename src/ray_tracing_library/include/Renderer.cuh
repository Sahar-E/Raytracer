//
// Created by Sahar on 10/06/2022.
//

#pragma once

#include "cuda_runtime_api.h"
#include <vector>
#include "World.cuh"
#include "Camera.cuh"

class Renderer {

public:

    Renderer(int imageWidth, int imageHeight, const World &world, const Camera &camera, int rayBounces);

    Renderer() = delete;
    Renderer(const Renderer &other) = delete;
    Renderer &operator=(const Renderer &other) = delete;

    virtual ~Renderer();

    void render();

    const Color * getPixelsOut() const;
    int getNPixelsOut() const { return _imgH * _imgW; }

    void setCamera(const Camera &camera);
    void clearPixels();

private:
    int _imgW;
    int _imgH;
    int _sharedMemForSpheres;
    World** _d_world;
    Camera _camera;
    const int _nRayBounces;
    float _alreadyNPixelsGot{};
    curandState *_randStates;
    Color *_pixelsOut{};
    Ray *_rays{};

    static World **allocateWorldInDeviceMemory(const Sphere *ptrSpheres, size_t nSpheres);
    static void freeWorldFromDeviceAndItsPtr(World **d_world);
    void initRandStates();

    void initPixelAllocations();
};



