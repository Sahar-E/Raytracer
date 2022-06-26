//
// Created by Sahar on 08/06/2022.
//

#pragma once

#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include <math_constants.h>
#include <cmath>
#include "constants.h"


/**
 * @return a random real in [0,1).
 */
__host__ __device__ inline double randomDouble(int &randState) {
    randState = (randState ^ 61) ^ (randState >> 16);
    randState = randState + (randState << 3);
    randState = randState ^ (randState >> 4);
    randState = randState * 0x27d4eb2d;
    randState = randState ^ (randState >> 15);
    return (double) randState / INT_MAX;
//    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
//    static std::minstd_rand generator(1);
//    return distribution(generator);
}

__host__ __device__ inline double randomDouble(int &randState, double from, double to) {
    return randomDouble(randState) * (from - to) + from;
}

__host__ __device__ inline double deg2rad(double degree) {
    return degree * CUDART_PI_F / 180.0;
}

/**
* Floating point comparison.
*
* @param a     First floating point value.
* @param b     Second floating point value.
* @return      true if the absolute difference between the two values is less than EPS.
*/
__host__ __device__ inline bool fcmp(double a, double b) {
    return fabs(a - b) < EPS;
}


/**
 * Clamp the number between 2 values.
 * @param toClamp   Number to clamp.
 * @param low       Low bound.
 * @param high      High bound.
 * @return  Clamp result.
 */
__host__ __device__ inline double clamp(double toClamp, double low, double high) {
    if (toClamp < low) {
        return low;
    }
    if (toClamp > high) {
        return high;
    }
    return toClamp;
}


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
__host__ __device__ inline bool cannotRefractBySnellsLaw(double cosTheta, double refractionIdxRatio) {
    double sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    bool cannotRefract = refractionIdxRatio * sinTheta > 1.0;
    return cannotRefract;
}


/**
 * Check if shouldn't refract by using Schlick's approximation for Frensel equations.
 *
 * @param cosTheta              The cosine of the angle theta. The angle between the normal and the incoming ray.
 * @param refractionIdxRatio    Refraction index of first material divided by the second.
 * @return  If should ref
 */
__host__ __device__ inline double reflectSchlickApproxForFrensel(double cosTheta, double refractionIdxRatio) {
    double r0 = (1 - refractionIdxRatio) / (1 + refractionIdxRatio);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow(1 - cosTheta, 5);
}
