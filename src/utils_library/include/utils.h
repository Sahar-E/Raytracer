//
// Created by Sahar on 08/06/2022.
//

#pragma once

#include <vector>
#include "Vec3.h"

/**
 * Save image to JPG file.
 *
 * @param filename      Name of the JPG.
 * @param data          Pixel data of the image.
 * @param width         Width of the image.
 * @param height        Height of the image.
 * @param channelCount  3 for RGB.
 */
void saveImgAsJpg(const std::string &filename,
                  const std::vector<Vec3> &data,
                  int width,
                  int height,
                  int channelCount);


/**
 * Performs alpha blending between 2 colors.
 *
 * @param v1        First color.
 * @param v2        Second color.
 * @param alpha     Ratio of colors. (e.g. 1 will be only c1)
 * @return  new color.
 */
Color alphaBlending(Color v1, Color v2, double alpha);


/**
* Floating point comparison.
*
* @param a     First floating point value.
* @param b     Second floating point value.
* @return      true if the absolute difference between the two values is less than EPS.
*/
bool fcmp(double a, double b);


/**
 * Do gamma correction (of 2.0) to the given color.
 * @param color The color to do gamma correction to.
 * @return  Corrected color.
 */
Color gammaCorrection(Color const &color);


/**
 * Clamp the number between 2 values.
 * @param toClamp   Number to clamp.
 * @param low       Low bound.
 * @param high      High bound.
 * @return  Clamp result.
 */
double clamp(double toClamp, double low, double high);

/**
 * Clamp the vector between 2 values.
 * @param toClamp   Number to clamp.
 * @param low       Low bound.
 * @param high      High bound.
 * @return  Clamp result.
 */
Vec3 clamp(const Vec3 &toClamp, double low, double high);

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
bool cannotRefractBySnellsLaw(double cosTheta, double refractionIdxRatio);

/**
 * Check if shouldn't refract by using Schlick's approximation for Frensel equations.
 *
 * @param cosTheta              The cosine of the angle theta. The angle between the normal and the incoming ray.
 * @param refractionIdxRatio    Refraction index of first material divided by the second.
 * @return  If should ref
 */
double reflectSchlickApproxForFrensel(double cosTheta, double refractionIdxRatio);
