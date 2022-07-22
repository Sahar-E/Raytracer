//
// Created by Sahar on 10/06/2022.
//

#pragma once

#include "cuda_runtime_api.h"
#include <vector>
#include <memory>
#include "World.cuh"
#include "Camera.cuh"

class RayTracerRenderer {

public:

    RayTracerRenderer(int imageWidth, int imageHeight, const World &world, std::shared_ptr<Camera> camera, int rayBounces);

    RayTracerRenderer() = delete;
    RayTracerRenderer(const RayTracerRenderer &other) = delete;
    RayTracerRenderer &operator=(const RayTracerRenderer &other) = delete;

    virtual ~RayTracerRenderer();

    void render();

    const Color * getPixelsOut() const;
    std::shared_ptr<unsigned char[]> getPixelsOutAsChars();
    int getNPixelsOut() const { return _imgH * _imgW; }

    void setCamera(std::shared_ptr<Camera> camera);
    void clearPixels();
    void syncPixelsOut();

    int getImgW() const;

    int getImgH() const;

private:
    int _imgW;
    int _imgH;
    int _sharedMemForSpheres;
    World** _d_world;
    std::shared_ptr<Camera> _camera;
    const int _nRayBounces;
    float _alreadyNPixelsGot{};
    curandState *_randStates;
    Color *_pixelsOut{};
    Color *_pixelsAverage{};
    std::shared_ptr<unsigned char[]> _pixelsOutAsChars{};
    std::shared_ptr<Color []> _pixelsOutAsColors{};
    Ray *_rays{};

    static World **allocateWorldInDeviceMemory(const Sphere *ptrSpheres, size_t nSpheres);
    static void freeWorldFromDeviceAndItsPtr(World **d_world);
    void initRandStates();

    void initPixelAllocations();

};



