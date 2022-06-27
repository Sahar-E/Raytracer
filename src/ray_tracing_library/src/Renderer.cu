//
// Created by Sahar on 10/06/2022.
//

#include <vector>
#include <Renderer.cuh>
#include "utils.cuh"
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
//         Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}


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
void getPixel(Color *pixel,
              Camera c,
              World **d_world,
              int nSamplesPerPixel,
              int row,
              int col,
              int *randState,
              int imWidth,
              int imHeight,
              int nBounces) {
    Color pixelSum{0,0,0};
    for (int i = 0; i < nSamplesPerPixel; ++i) {
        auto h = (static_cast<double>(col) + randomDouble(*randState)) / (imWidth - 1);
        auto v = 1 - ((static_cast<double>(row) + randomDouble(*randState)) / (imHeight - 1));
        Ray ray = c.getRay(h, v, *randState);
        pixelSum += (*d_world)->rayTrace(ray, nBounces, *randState);
    }
    *pixel = pixelSum;
}

void getPixel2(Color &pixel,
               const Camera &c,
               const World &d_world,
               int nSamplesPerPixel,
               int row,
               int col,
               int *randState,
               int imWidth,
               int imHeight,
               int nBounces) {
    Color pixelSum{0,0,0};
    for (int i = 0; i < nSamplesPerPixel; ++i) {
        auto h = (static_cast<double>(col) + randomDouble(*randState)) / (imWidth - 1);
        auto v = 1 - ((static_cast<double>(row) + randomDouble(*randState)) / (imHeight - 1));
        Ray ray = c.getRay(h, v, *randState);
        pixelSum += d_world.rayTrace(ray, nBounces, *randState);
    }
    pixel = pixelSum / nSamplesPerPixel;
}

std::vector<Color> Renderer::render() const {
    World **d_world = allocateWorldInDeviceMemory(_world.getSpheres(), _world.getNSpheres());

    std::vector<Color> data(_imageHeight * _imageWidth, {1, 1, 1});
    for (int pixel_idx = 0; pixel_idx < _imageHeight * _imageWidth; ++pixel_idx) {
        int row = pixel_idx / _imageWidth;
        int col = pixel_idx % _imageWidth;

        Color *pixelRes;
        int *randState;
        checkCudaErrors(cudaMallocManaged(&pixelRes, sizeof(Color) * 1));
        checkCudaErrors(cudaMallocManaged(&randState, sizeof(int) * 1));


        getPixel<<<1, 1>>>(pixelRes, _camera, d_world, _nSamplesPerPixel, row, col, randState, _imageWidth, _imageHeight, _nRayBounces);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        Vec3 pixelColor = *pixelRes;
        checkCudaErrors(cudaFree(pixelRes));
        checkCudaErrors(cudaFree(randState));

//        // TODO-Sahar: Remove:
//        Vec3 pixelColor{};
//        int demoRand = 0;
//        getPixel2(pixelColor, _camera, _world, _nSamplesPerPixel, row, col, &demoRand, _imageWidth, _imageHeight, _nRayBounces);

        pixelColor = clamp(gammaCorrection(pixelColor), 0.0, 0.999);
        data[row * _imageWidth + col] = pixelColor;
    }
    freeWorldFromDeviceAndItsPtr(d_world);
    return data;
}

void Renderer::freeWorldFromDeviceAndItsPtr(World **d_world) {
    // Free world object from the device.
    freeWorld<<<1,1>>>(d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Free the pointer to the world object.
    checkCudaErrors(cudaFree(d_world));
}

World **Renderer::allocateWorldInDeviceMemory(const Sphere *ptrSpheres, size_t nSpheres) {
    // Copy the sphereArr to the GPU memory.
    Sphere * sphereArr;
    checkCudaErrors(cudaMallocManaged((void **) &sphereArr, sizeof(Sphere) * nSpheres));
    for (int i = 0; i < nSpheres; ++i) {
        sphereArr[i] = ptrSpheres[i];
    }

    // Create d_world with the sphereArr.
    World **d_world;
    checkCudaErrors(cudaMallocManaged(&d_world, sizeof(World *) * 1));
    createWorld<<<1,1>>>(d_world, sphereArr, nSpheres);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Free the sphereArr memory from the GPU memory.
    checkCudaErrors(cudaFree(sphereArr));
    return d_world;
}
