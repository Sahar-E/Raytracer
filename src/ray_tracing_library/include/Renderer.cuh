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
    int getNPixelsOut() const { return _imageHeight * _imageWidth; }

private:
    int _imageWidth;
    int _imageHeight;
    World** _d_world;
    Camera _camera;
    const int _nRayBounces;
    int _nSamplesPerPixel;
    curandState *_randStates;
    Color *_pixelsOut{};
    Color *_renderedPixels{};

    static World **allocateWorldInDeviceMemory(const Sphere *ptrSpheres, size_t nSpheres);
    static void freeWorldFromDeviceAndItsPtr(World **d_world);
    void initRandStates();

    void initPixelAllocations();
};



