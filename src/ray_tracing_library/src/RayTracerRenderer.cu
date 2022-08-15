//
// Created by Sahar on 10/06/2022.
//

#include <utility>
#include <vector>
#include "RayTracerRenderer.cuh"
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
void traceRay(Color *pixelsOut, Color *pixelsAverageAccum, Camera c, World const *const *d_world, curandState *randStates,
         const int imWidth, const int imHeight, const int nBounces, const float alreadyNPixelsGot) {
    // TODO-Sahar: Profile again with NSight, and then refactor this function.
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
            attenuationColors[hitCount] = {0.0f, 0.0f, 0.0f};
            emittedColors[hitCount] = {0.0f, 0.0f, 0.0f};
            hitCount++;
        }

        // Unrolling back the results in the stacks.
        Color res = {1.0f,1.0f,1.0f};
        for(int i = hitCount-1; i >= 0; --i) {
            res = emittedColors[i] + attenuationColors[i] * res;
        }
        pixelsAverageAccum[tIdx] = (pixelsAverageAccum[tIdx] + (res - pixelsAverageAccum[tIdx]) / alreadyNPixelsGot);
        pixelsOut[tIdx] = clamp(gammaCorrection(pixelsAverageAccum[tIdx]), 0.0f, 0.999f) * 255.0f;;
        randStates[tIdx] = localRandState;
    }
}

void RayTracerRenderer::render() {
    ++_alreadyNPixelsGot;
    int nPixels = _imgH * _imgW;
    int blockSize = BLOCK_SIZE;
    int numBlocks = (nPixels + blockSize - 1) / blockSize;
    traceRay<<<numBlocks, blockSize, _d_SharedMemForSpheres>>>(_pixelsOut_cuda, _innerPixelsAverageAccum_cuda,
                                                               *_camera, _d_world, _randStates, _imgW, _imgH,
                                                               _nRayBounces, _alreadyNPixelsGot);
}

void RayTracerRenderer::freeWorldFromDeviceAndItsPtr(World **d_world) {
    // Free world object from the device.
    freeWorld<<<1, 1>>>(d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Free the pointer to the world object.
    checkCudaErrors(cudaFree(d_world));
}


World **RayTracerRenderer::allocateWorldInDeviceMemory(const Sphere *ptrSpheres, size_t nSpheres) {
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


RayTracerRenderer::RayTracerRenderer(const int imageWidth, const int imageHeight, const World &world, std::shared_ptr<Camera> camera,
                                     int rayBounces) : _imgW(imageWidth), _imgH(imageHeight),
                                                       _d_SharedMemForSpheres(world.getTotalSizeInMemoryForObjects()),
                                                       _d_world(),
                                                       _camera(std::move(camera)),
                                                       _randStates(),
                                                       _nRayBounces(rayBounces),
                                                       _alreadyNPixelsGot(0.0f),
                                                       _pixelsOutAsChars(std::shared_ptr<unsigned char[]>(new unsigned char[_imgH * _imgW * sizeof(float)])),
                                                       _pixelsOutAsColors(std::shared_ptr<Color []>(new Color[_imgH * _imgW])){
    _d_world = allocateWorldInDeviceMemory(world.getSpheres(), world.getNSpheres());

    initRandStates();
    initImgBuffersAllocations();
}

void RayTracerRenderer::initImgBuffersAllocations() {
    int nPixels = _imgW * _imgH;
    checkCudaErrors(cudaMallocManaged(&_pixelsOut_cuda, sizeof(Color) * nPixels));
    checkCudaErrors(cudaMallocManaged(&_innerPixelsAverageAccum_cuda, sizeof(Color) * nPixels));
    checkCudaErrors(cudaMallocManaged(&_rays, sizeof(Ray) * nPixels));
    for (int i = 0; i < nPixels; ++i) {
        _pixelsOut_cuda[i] = {0, 0, 0};
        _innerPixelsAverageAccum_cuda[i] = {0, 0, 0};
        _rays[i] = {{0, 0, 0}, {0,0,0}};
    }
}

void RayTracerRenderer::initRandStates() {
    int nPixels = _imgW * _imgH;
    int nThreads = BLOCK_SIZE;
    size_t blockSize = (nPixels + nThreads - 1) / nThreads;
    checkCudaErrors(cudaMalloc(&_randStates, sizeof(curandState) * nPixels));
    initCurand<<<blockSize, nThreads>>>(_randStates, nPixels, 0);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

RayTracerRenderer::~RayTracerRenderer() {
    checkCudaErrors(cudaFree(_rays));
    checkCudaErrors(cudaFree(_pixelsOut_cuda));
    checkCudaErrors(cudaFree(_innerPixelsAverageAccum_cuda));
    checkCudaErrors(cudaFree(_randStates));
    freeWorldFromDeviceAndItsPtr(_d_world);
}


static void copyRGBToCharArray(std::shared_ptr<unsigned char[]> &pixelsOutAsChars, const Color *pixelsOut, int nPixels) {
    int channelCount = 3;
    for (int i = 0; i < nPixels; ++i) {
        Color pixel = pixelsOut[i];
        pixelsOutAsChars[i * channelCount] =     static_cast<unsigned char>(pixel.x());
        pixelsOutAsChars[i * channelCount + 1] = static_cast<unsigned char>(pixel.y());
        pixelsOutAsChars[i * channelCount + 2] = static_cast<unsigned char>(pixel.z());
    }
}

std::shared_ptr<unsigned char[]> RayTracerRenderer::getPixelsOutAsChars() const {
    return _pixelsOutAsChars;
}

void RayTracerRenderer::syncPixelsOutAsChars() {
    size_t sizeOfCopy = _imgH * _imgW * sizeof(Color);
    checkCudaErrors(cudaMemcpy(_pixelsOutAsColors.get(), _pixelsOut_cuda, sizeOfCopy, cudaMemcpyDefault));
    copyRGBToCharArray(_pixelsOutAsChars, _pixelsOutAsColors.get(), _imgH * _imgW);
}

void RayTracerRenderer::clearPixels() {
    int nPixels = _imgW * _imgH;
    checkCudaErrors(cudaMemset(_pixelsOut_cuda, 0, nPixels * sizeof(Color)));
    _alreadyNPixelsGot = 0;
}

int RayTracerRenderer::getImgW() const {
    return _imgW;
}

int RayTracerRenderer::getImgH() const {
    return _imgH;
}

int RayTracerRenderer::getNRayBounces() const {
    return _nRayBounces;
}

void RayTracerRenderer::setNRayBounces(int nRayBounces) {
    _nRayBounces = nRayBounces;
    clearPixels();
}
