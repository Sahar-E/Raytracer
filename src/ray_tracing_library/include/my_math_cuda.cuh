//
// Created by Sahar on 08/06/2022.
//

#pragma once

#include <cstdint>
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>


/**
 * @return a random real in [0,1).
 */

__device__ float randomFloatCuda(curandState *state);

__global__ void initCurand(curandState *randStates, uint32_t n_randStates, uint32_t seed);

__host__ __device__ double deg2rad(double degree);

/**
* Floating point comparison.
*
* @param a     First floating point value.
* @param b     Second floating point value.
* @return      true if the absolute difference between the two values is less than 1e-6.
*/
__host__ __device__ bool fcmp(double a, double b);


/**
 * Clamp the number between 2 values.
 * @param toClamp   Number to clamp.
 * @param low       Low bound.
 * @param high      High bound.
 * @return  Clamp result.
 */
__host__ __device__ double clamp(double toClamp, double low, double high);


/**
 * Check if it is possible to refract the ray according to the cosine of the angle between the normal and the
 * incoming ray, and using the refractionIdxRatio that is the previous material refractionIdx divided by the target
 * refractionIdx of the target material.
 *
 * Snell's Law is:
 * sin(theta1) / sin(theta2)  =  eta1 / eta2
 *
 * <==> sin(theta1) = (eta1 / eta2) * sin(theta2)
 *
 *
 * Reminder that sin(theta1) is bounded by [-1,1].
 *
 * @param cosTheta              The cosine of the angle theta1. (Reminder that cos(angle) is equal to the dot product
 *                                  between the two vectors, if they have the same origin and are normalized)
 * @param refractionIdxRatio    Refraction index of first material divided by the second.
 * @return  If possible to refract according to Snell's Law.
 */
__host__ __device__ bool cannotRefractBySnellsLaw(double cosTheta, double refractionIdxRatio);


/**
 * Check if shouldn't refract by using Schlick's approximation for Frensel equations.
 *
 * @param cosTheta              The cosine of the angle theta. The angle between the normal and the incoming ray.
 * @param refractionIdxRatio    Refraction index of first material divided by the second.
 * @return  If should ref
 */
__host__ __device__ double reflectSchlickApproxForFrensel(double cosTheta, double refractionIdxRatio);
