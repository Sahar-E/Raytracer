#include <iostream>
#include <vector>
#include <World.cuh>
#include <Camera.cuh>
#include <Renderer.cuh>
#include "utils.cuh"
#include "TimeThis.h"
#include "commonDefines.h"
#include <string>
#include <cassert>
#include "cuda_runtime_api.h"
#include "commonCuda.cuh"

//__global__ void test2d(int n, Ray *rays, int nBounces, World **d_world, curandState *randStates) {
//    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
//    printf("blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d, gridDim.x: %d\n", blockIdx.x, blockDim.x, threadIdx.x,  gridDim.x);
//    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (pixel_idx < n) {
//        size_t n_spheres = (*d_world)->getNSpheres();
//        Sphere* spheres = (*d_world)->getSpheres();
//        auto curRay(rays[pixel_idx]);
//        int hitCount = 0;
//        while (hitCount < nBounces) {
//            for (int i = 0; i < n_spheres; ++i) {
//                // Do something...
//            }
//            // with early break
//        }
//
//}


int main() {
//    TimeThis t("main");
//    float *arr;
//    int n = 50;
//    checkCudaErrors(cudaMallocManaged(&arr, sizeof(float) * n));
//    for (int i = 0; i < n; ++i) {
//        arr[i] = 1;
//    }
//
//    int n_inner_loops = 10;
//    for (int i = 0; i < n / n_inner_loops; ++i) {
//        for (int j = 0; j < n_inner_loops; ++j) {
//            arr[i*10+j] = 1;
//        }
//    }
//
//
//    int gridSize = 10;
//    int blockSize = (n + gridSize - 1) / gridSize;
//    test2d<<<blockSize,gridSize>>>();
//    checkCudaErrors(cudaGetLastError());
//    checkCudaErrors(cudaDeviceSynchronize());
//
//    checkCudaErrors(cudaFree(arr));




    const auto aspectRatio = 3.0f / 2.0f;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspectRatio);
    const int rayBounces = 7;
    int vFov = 26.0f;
    float aperture = 0.05f;
    float focusDist = 10.0f;
    int nFrames = 1500;
//    int nFrames = 1;

    assert(0 < rayBounces && rayBounces <= MAX_BOUNCES);

    Vec3 vUp = {0, 1, 0};
    Vec3 lookFrom = {0, 1.8, 12};
    Vec3 lookAt = {0., 0, 0};


    auto world = World::initWorld2();
    std::cout << "Size: " << world.getTotalSizeInSharedMemory()  << "\n";
    std::cout << "nSpheres: " << world.getNSpheres()  << "\n";
    assert(world.getTotalSizeInSharedMemory() < 48 * pow(2, 10) && "There is a hard limit for NVIDIA's shared memory size of 48KB for one block.");
    auto camera = Camera(lookFrom, lookAt, vUp, aspectRatio, vFov, aperture, focusDist);
    Renderer renderer(image_width, image_height, world, camera, rayBounces);

    for (int j = 0; j < nFrames; ++j) {
        renderer.render();
        std::cout << "Done iteration #: " << j  << "\n";
    }

    std::string filename = "test.jpg";
    int channelCount = 3;
    std::vector<std::tuple<float, float, float>> rgb(renderer.getNPixelsOut(), {0, 0, 0});
    for (int i = 0; i < renderer.getNPixelsOut(); ++i) {
        Color pixel = renderer.getPixelsOut()[i];
        pixel = clamp(gammaCorrection(pixel), 0.0, 0.999);
        rgb[i] = {pixel.x(), pixel.y(), pixel.z()};
    }
    saveImgAsJpg(filename, rgb, image_width, image_height, channelCount);

    std::cout << "Done." << "\n";
    return 0;
}