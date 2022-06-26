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
              World *d_world,
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
        pixelSum += d_world->rayTrace(ray, nBounces, *randState);
    }
    *pixel = pixelSum;
}

void getPixel2(Color *pixel,
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

std::vector<Color> Renderer::render() const {
    std::vector<Color> data(_imageHeight * _imageWidth, {1, 1, 1});
    for (int pixel_idx = 0; pixel_idx < _imageHeight * _imageWidth; ++pixel_idx) {
        int row = pixel_idx / _imageWidth;
        int col = pixel_idx % _imageWidth;
        Color *pixelRes;
        int *randState;
//        pixelRes = new Color;
//        randState = new int;
//        World **d_world;
        checkCudaErrors(cudaMallocManaged(&randState, sizeof(int) * 1));
        checkCudaErrors(cudaMallocManaged(&pixelRes, sizeof(Color) * 1));
//        checkCudaErrors(cudaMallocManaged(&d_world, sizeof(void **) * 1));
        World *d_world;
        Sphere * sphereArr;
        checkCudaErrors(cudaMallocManaged(&d_world, sizeof(World) * 1));
        size_t nSpheres = _world.getNSpheres();
        checkCudaErrors(cudaMallocManaged(&sphereArr, sizeof(Sphere) * nSpheres));
        for (int i = 0; i < nSpheres; ++i) {
            sphereArr[i] = _world.getSpheres()[i];
        }
        *d_world = World(sphereArr, nSpheres);

        getPixel<<<1, 1>>>(pixelRes, _camera, d_world, _nSamplesPerPixel, row, col, randState, _imageWidth, _imageHeight, _nRayBounces);


        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(d_world));
        checkCudaErrors(cudaFree(sphereArr));


//        createWorld<<<1, 1>>>(d_world, _world.getObjects().getSpheres(), _world.getObjects().getNSpheres());
//        checkCudaErrors(cudaGetLastError());
//        checkCudaErrors(cudaDeviceSynchronize());
//        *randState = 0;
//        *pixelRes = {0, 0, 0};

//        getPixel<<<1, 1>>>(pixelRes, _camera, d_world, _nSamplesPerPixel, row, col, randState, _imageWidth, _imageHeight, _nRayBounces);
//        checkCudaErrors(cudaGetLastError());
//        checkCudaErrors(cudaDeviceSynchronize());

//        getPixel2(pixelRes, _camera, _world, _nSamplesPerPixel, row, col, randState, _imageWidth, _imageHeight, _nRayBounces);

        Vec3 pixelColor = *pixelRes;
//        Vec3 pixelColor = {0,0,0};
//
        checkCudaErrors(cudaFree(pixelRes));
        checkCudaErrors(cudaFree(randState));
//        freeWorld<<<1,1>>>(d_world);
//        checkCudaErrors(cudaGetLastError());
//        checkCudaErrors(cudaFree(d_world));

//        delete pixelRes;
//        delete randState;

        pixelColor = clamp(gammaCorrection(pixelColor), 0.0, 0.999);
        data[row * _imageWidth + col] = pixelColor;
    }
    return data;
}
