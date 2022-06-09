//
// Created by Sahar on 08/06/2022.
//

#include <string>
#include <vector>
#include <memory>
#include <stb_library/stb_image_write.h>
#include "utils.h"

void saveImgAsJpg(const std::string &filename,
                  const std::vector<Vec3<>> &data,
                  const int width,
                  const int height,
                  const int channelCount) {
    int n = height * width;
    auto dataCopy = std::make_unique<unsigned char[]>(n*channelCount);
    for (int i = 0; i < n; ++i) {
        dataCopy[i*channelCount] = static_cast<unsigned char>(data[i].x() * 255);
        dataCopy[i*channelCount + 1] = static_cast<unsigned char>(data[i].y() * 255);
        dataCopy[i*channelCount + 2] = static_cast<unsigned char>(data[i].z() * 255);
    }
    std::cout << "\nSaving image to " << filename << "\n";
    stbi_write_jpg(filename.c_str(), width, height, channelCount, dataCopy.get(), width * channelCount);
}