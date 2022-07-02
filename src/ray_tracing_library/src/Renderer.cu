//
// Created by Sahar on 10/06/2022.
//

#include <vector>
#include "Renderer.cuh"
#include "utils.cuh"
#include "TimeThis.h"
#include "commonCuda.cuh"
#include "commonDefines.h"
#include <iostream>
#include <cuda_runtime_api.h>

__global__
void createWorld(World **deviceWorld, Sphere *spheres, size_t nSpheres) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *deviceWorld = new World(spheres, nSpheres);
    }
}


__global__
void freeWorld(World **deviceWorld) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *deviceWorld;
    }
}

__global__
void writePixels(Color *pixelsOut, Camera c, World **d_world, curandState *randStates, int imWidth, int imHeight,
                 int nBounces) {
    size_t n_spheres = (*d_world)->getNSpheres();
    Sphere* spheres = (*d_world)->getSpheres();
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx < imHeight * imWidth) {
        curandState localRandState = randStates[pixel_idx];
        int row = pixel_idx / imWidth;
        int col = pixel_idx % imWidth;
        auto h = (static_cast<double>(col) + randomFloatCuda(&localRandState)) / (imWidth - 1);
        auto v = 1 - ((static_cast<double>(row) + randomFloatCuda(&localRandState)) / (imHeight - 1));
        Ray ray = c.getRay(h, v, &localRandState);
        // TODO-Sahar: can pack up all the results in shared memory and in one swoosh return it.
        pixelsOut[pixel_idx] = World::rayTrace(spheres, n_spheres, ray, nBounces, &localRandState);
        randStates[pixel_idx] = localRandState;
    }

}

__global__
void getAverageForPixels(Color *pixelsAverageNew, Color *pixelsAverageOld, Color *pixels, size_t n, int n_pixelsAlreadySummed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        pixelsAverageNew[index] = pixelsAverageOld[index] +
                                  (pixels[index] - pixelsAverageOld[index]) / static_cast<float>(n_pixelsAlreadySummed);
    }
}


void Renderer::render() {
    TimeThis t("render");
    int nPixels = _imageHeight * _imageWidth;

    int blockSize = N_THREAD;
    int numBlocks = (nPixels + blockSize - 1) / blockSize;
    writePixels<<<numBlocks, blockSize>>>(_renderedPixels, _camera, _d_world, _randStates,
                                          _imageWidth, _imageHeight, _nRayBounces);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    getAverageForPixels<<<numBlocks, blockSize>>>(_pixelsOut, _pixelsOut, _renderedPixels, nPixels, ++_nSamplesPerPixel);   // TODO-Sahar: 60ms - slow.
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void Renderer::freeWorldFromDeviceAndItsPtr(World **d_world) {
    // Free world object from the device.
    freeWorld<<<1, 1>>>(d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Free the pointer to the world object.
    checkCudaErrors(cudaFree(d_world));
}


World **Renderer::allocateWorldInDeviceMemory(const Sphere *ptrSpheres, size_t nSpheres) {
    // Copy the sphereArr to the GPU memory.
    Sphere *sphereArr;
    checkCudaErrors(cudaMallocManaged((void **) &sphereArr, sizeof(Sphere) * nSpheres));
    for (int i = 0; i < nSpheres; ++i) {
        sphereArr[i] = ptrSpheres[i];
    }

    // Create d_world with the sphereArr.
    World **d_world;
    checkCudaErrors(cudaMallocManaged(&d_world, sizeof(World *) * 1));
    createWorld<<<1, 1>>>(d_world, sphereArr, nSpheres);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Free the sphereArr memory from the GPU memory.
    checkCudaErrors(cudaFree(sphereArr));
    return d_world;
}


Renderer::Renderer(const int imageWidth, const int imageHeight, const World &world, const Camera &camera,
                   int rayBounces) : _imageWidth(imageWidth), _imageHeight(imageHeight),
                                                           _d_world(),
                                                           _camera(camera),
                                                           _randStates(),
                                                           _nRayBounces(rayBounces),
                                                           _nSamplesPerPixel(0) {
    _d_world = allocateWorldInDeviceMemory(world.getSpheres(), world.getNSpheres());

    initRandStates();
    initPixelAllocations();
}

void Renderer::initPixelAllocations() {
    int nPixels = _imageWidth * _imageHeight;
    checkCudaErrors(cudaMallocManaged(&_pixelsOut, sizeof(Color) * nPixels));
    checkCudaErrors(cudaMallocManaged(&_renderedPixels, sizeof(Color) * nPixels));
    for (int i = 0; i < nPixels; ++i) {
        _pixelsOut[i] = {0, 0, 0};
        _renderedPixels[i] = {0, 0, 0};
    }
}

void Renderer::initRandStates() {
    int nPixels = _imageWidth * _imageHeight;
    int nThreads = N_THREAD;
    size_t blockSize = (nPixels + nThreads - 1) / nThreads;
    checkCudaErrors(cudaMalloc(&_randStates, sizeof(curandState) * nPixels));
    initCurand<<<blockSize, nThreads>>>(_randStates, nPixels, 0);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

Renderer::~Renderer() {
    checkCudaErrors(cudaFree(_renderedPixels));
    checkCudaErrors(cudaFree(_pixelsOut));
    checkCudaErrors(cudaFree(_randStates));
    freeWorldFromDeviceAndItsPtr(_d_world);
}

const Color * Renderer::getPixelsOut() const {
    return _pixelsOut;
}

World **allocateWorldInDeviceMemory2(const Sphere *ptrSpheres, size_t nSpheres) {
    auto **d_world = new World*;
    *d_world = new World(ptrSpheres, nSpheres);
    return d_world;
}
