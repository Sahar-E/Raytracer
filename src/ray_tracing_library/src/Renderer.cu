//
// Created by Sahar on 10/06/2022.
//

#include <vector>
#include "Renderer.cuh"
#include "utils.h"
#include "TimeThis.h"
#include "commonCuda.cuh"
#include "commonDefines.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include <cfloat>

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
void traceRay(Color *pixelsOut, Camera c  , World const *const *d_world, curandState *randStates,
              const int imWidth, const int imHeight, const int nBounces, const float alreadyNPixelsGot) {
    // TODO-Sahar: Still work in progress...
    extern __shared__ Sphere spheresArr[];
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int nPixels = imHeight * imWidth;
    if (tIdx < nPixels) {

        size_t n_spheres = (*d_world)->getNSpheres();
        Sphere* spheres = (*d_world)->getSpheres();

        // Copy to the shared memory all the spheres for each block.
        for (int i = threadIdx.x; i < n_spheres; i += blockDim.x) {
            spheresArr[i] = spheres[i];
        }

        curandState localRandState = randStates[tIdx];

        // Get Ray:
        int row = tIdx / imWidth;
        int col = tIdx % imWidth;
        auto h = (static_cast<float>(col) + randomFloatCuda(&localRandState)) / (imWidth - 1.0f);
        auto v = 1.0f - ((static_cast<float>(row) + randomFloatCuda(&localRandState)) / (imHeight - 1.0f));
        auto curRay = c.getRay(h, v, &localRandState);

        Color attenuationColors[MAX_BOUNCES];
        Color emittedColors[MAX_BOUNCES];
        int hitCount = 0;

        __syncthreads();    // For the spheresArr initialization.

        // Iterative ray tracing:
        while (hitCount < nBounces) {
            bool hit = false;
            float tEnd = FLT_MAX;
            int hitSphereIdx = -1;
            float rootRes;
            for (int i = 0; i < n_spheres; ++i) {
                if (spheresArr[i].isHit(curRay, CLOSEST_POSSIBLE_RAY_HIT, tEnd, rootRes)) {
                    hit = true;
                    tEnd = rootRes;
                    hitSphereIdx = i;
                }
            }
            if (hit) {
                HitResult hitRes;
                spheresArr[hitSphereIdx].getHitResult(curRay, rootRes, hitRes);

                Color emittedColor{}, attenuation{};
                spheresArr[hitSphereIdx].getMaterial().getColorAndSecondaryRay(hitRes, &localRandState,
                                                                            emittedColor, attenuation, curRay);
                attenuationColors[hitCount] = attenuation;
                emittedColors[hitCount] = emittedColor;
                ++hitCount;
            } else {
                attenuationColors[hitCount] = World::backgroundColor(curRay);
                emittedColors[hitCount] = {0.0f, 0.0f, 0.0f};
                ++hitCount;
                break;
            }
        }
        if (hitCount == nBounces) {
            attenuationColors[hitCount-1] = {0.0f, 0.0f, 0.0f};
            emittedColors[hitCount-1] = {0.0f, 0.0f, 0.0f};
        }

        // Unrolling back the results in the stacks.
        Color res = {1.0f,1.0f,1.0f};
        for(int i = hitCount-1; i >= 0; --i) {
            res = emittedColors[i] + attenuationColors[i] * res;
        }

        pixelsOut[tIdx] = pixelsOut[tIdx] + (res - pixelsOut[tIdx]) / alreadyNPixelsGot;
        randStates[tIdx] = localRandState;
    }

}

void Renderer::render() {
    TimeThis t("render");
    int nPixels = _imgH * _imgW;

    int blockSize = BLOCK_SIZE;
    int numBlocks = (nPixels + blockSize - 1) / blockSize;
    ++_alreadyNPixelsGot;
    traceRay<<<numBlocks, blockSize, _sharedMemForSpheres>>>(_pixelsOut, _camera, _d_world, _randStates, _imgW, _imgH,
                                                             _nRayBounces, _alreadyNPixelsGot);
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
                   int rayBounces) : _imgW(imageWidth), _imgH(imageHeight),
                                     _sharedMemForSpheres(world.getTotalSizeInMemoryForObjects()),
                                     _d_world(),
                                     _camera(camera),
                                     _randStates(),
                                     _nRayBounces(rayBounces),
                                     _alreadyNPixelsGot(0.0f) {
    _d_world = allocateWorldInDeviceMemory(world.getSpheres(), world.getNSpheres());

    initRandStates();
    initPixelAllocations();
}

void Renderer::initPixelAllocations() {
    int nPixels = _imgW * _imgH;
    checkCudaErrors(cudaMallocManaged(&_pixelsOut, sizeof(Color) * nPixels));
    checkCudaErrors(cudaMallocManaged(&_rays, sizeof(Ray) * nPixels));
    for (int i = 0; i < nPixels; ++i) {
        _pixelsOut[i] = {0, 0, 0};
        _rays[i] = {{0, 0, 0}, {0,0,0}};
    }
}

void Renderer::initRandStates() {
    int nPixels = _imgW * _imgH;
    int nThreads = BLOCK_SIZE;
    size_t blockSize = (nPixels + nThreads - 1) / nThreads;
    checkCudaErrors(cudaMalloc(&_randStates, sizeof(curandState) * nPixels));
    initCurand<<<blockSize, nThreads>>>(_randStates, nPixels, 0);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

Renderer::~Renderer() {
    checkCudaErrors(cudaFree(_rays));
    checkCudaErrors(cudaFree(_pixelsOut));
    checkCudaErrors(cudaFree(_randStates));
    freeWorldFromDeviceAndItsPtr(_d_world);
}

const Color * Renderer::getPixelsOut() const {
    return _pixelsOut;
}

void Renderer::setCamera(const Camera &camera) {
    _camera = camera;
}

void Renderer::clearPixels() {
    int nPixels = _imgW * _imgH;
    for (int i = 0; i < nPixels; ++i) {
        _pixelsOut[i] = {0, 0, 0};
    }
    _alreadyNPixelsGot = 0;
}

//std::shared_ptr<unsigned char[]> Renderer::getPixelsOutAsChars() const {
//
//    std::shared_ptr<unsigned char[]> ptr = std::shared_ptr<unsigned char[]>();
//    copyRGBToCharArray(ptr.get(), _pixelsOut);
//    return ptr;
//}
