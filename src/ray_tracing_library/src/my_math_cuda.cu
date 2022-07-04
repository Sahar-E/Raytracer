#include "my_math_cuda.cuh"
#include <iostream>
#include <cuda_runtime_api.h>
#include <math_constants.h>


__global__ void initCurand(curandState *randStates, uint32_t n_randStates, uint32_t seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_randStates) {
        curand_init(idx, 0, 0, &randStates[idx]);
    }
}

__device__ float randomFloatCuda(curandState *state) {
    return curand_uniform(state);
}

__host__ __device__ float deg2rad(float degree) {
    return degree * CUDART_PI_F / 180.0f;
}

__host__ __device__ bool fcmp(float a, float b) {
    return fabsf(a - b) < 1e-6f;
}

__host__ __device__ float clamp(float toClamp, float low, float high) {
    if (toClamp < low) {
        return low;
    }
    if (toClamp > high) {
        return high;
    }
    return toClamp;
}

__host__ __device__ bool cannotRefractBySnellsLaw(float cosTheta, float refractionIdxRatio) {
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    bool cannotRefract = refractionIdxRatio * sinTheta > 1.0f;
    return cannotRefract;
}

__host__ __device__ float reflectSchlickApproxForFrensel(float cosTheta, float refractionIdxRatio) {
    float r0 = (1.0f - refractionIdxRatio) / (1.0f + refractionIdxRatio);
    r0 = r0 * r0;
    float tmp = 1.0f - cosTheta;
    return r0 + (1.0f - r0) *
                tmp *
                tmp *
                tmp *
                tmp *
                tmp;  // 5 times.
}
