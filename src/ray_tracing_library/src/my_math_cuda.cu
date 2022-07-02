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

__host__ __device__ double deg2rad(double degree) {
    return degree * CUDART_PI_F / 180.0;
}

__host__ __device__ bool fcmp(double a, double b) {
    return fabs(a - b) < 1e-6;
}

__host__ __device__ double clamp(double toClamp, double low, double high) {
    if (toClamp < low) {
        return low;
    }
    if (toClamp > high) {
        return high;
    }
    return toClamp;
}

__host__ __device__ bool cannotRefractBySnellsLaw(double cosTheta, double refractionIdxRatio) {
    double sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    bool cannotRefract = refractionIdxRatio * sinTheta > 1.0;
    return cannotRefract;
}

__host__ __device__ double reflectSchlickApproxForFrensel(double cosTheta, double refractionIdxRatio) {
    double r0 = (1 - refractionIdxRatio) / (1 + refractionIdxRatio);
    r0 = r0 * r0;
    double tmp = 1 - cosTheta;
    return r0 + (1 - r0) *
                tmp *
                tmp *
                tmp *
                tmp *
                tmp;  // 5 times.
}
