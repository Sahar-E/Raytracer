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
 * @param c1        First color.
 * @param c2        Second color.
 * @param alpha     Ratio of colors. (e.g. 1 will be only c1)
 * @return  new color.
 */
Color alphaBlending(Color c1, Color c2, double alpha);


/**
* Floating point comparison.
*
* @param a     First floating point value.
* @param b     Second floating point value.
* @return      true if the absolute difference between the two values is less than EPS.
*/
bool fcmp(double a, double b);