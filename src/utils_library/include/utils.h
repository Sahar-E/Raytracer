//
// Created by Sahar on 08/06/2022.
//

#pragma once

#include <vector>
#include "Vec3.hpp"

const double BLACK_COLOR[3] = {0, 0, 0};
const double WHITE_COLOR[3] = {1, 1, 1};
const double SKY_COLOR[3] = {0.45, 0.6, 1.0};

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


Color alphaBlending(Color c1, Color c2, double alpha);