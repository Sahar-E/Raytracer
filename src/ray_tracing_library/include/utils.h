//
// Created by Sahar on 08/06/2022.
//

#pragma once

#include <vector>
#include <string>

/**
 * Save image to JPG file.
 *
 * @param filename  Name of the JPG.
 * @param data      Pixel data of the image.
 * @param width     Width of the image.
 * @param height    Height of the image.
 */
void saveImgAsJpg(const std::string &filename, const std::shared_ptr<unsigned char[]>& data, const int width,
                  const int height);
