//
// Created by Sahar on 08/06/2022.
//

#pragma once

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

