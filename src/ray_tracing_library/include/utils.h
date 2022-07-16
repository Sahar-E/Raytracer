//
// Created by Sahar on 08/06/2022.
//

#pragma once

#include <vector>
#include <string>

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
                  const std::vector<std::tuple<float, float, float>> &data,
                  const int width, const int height);

bool copyRGBToCharArray(unsigned char *dest, const std::vector<std::tuple<float, float, float>> &src, int n);