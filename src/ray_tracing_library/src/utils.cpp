//
// Created by Sahar on 08/06/2022.
//

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "stb_library/stb_image_write.h"
#include "utils.h"


bool copyRGBToCharArray(unsigned char *dest, const std::vector<std::tuple<float, float, float>>& src, int n_rgb) {
    int channelCount = 3;
    for (int i = 0; i < n_rgb; ++i) {
        auto [r, g, b] = src[i];
        dest[i * channelCount] =     static_cast<unsigned char>(r * 255);
        dest[i * channelCount + 1] = static_cast<unsigned char>(g * 255);
        dest[i * channelCount + 2] = static_cast<unsigned char>(b * 255);
    }
    return true;
}

void saveImgAsJpg(const std::string &filename, const std::vector<std::tuple<float, float, float>> &data, const int width,
             const int height) {
    int n = height * width;
    int channelCount = 3;
    auto dataCopy = std::make_unique<unsigned char[]>(n * channelCount);
    copyRGBToCharArray(dataCopy.get(), data, n);
    std::cout << "\nSaving image to " << filename << "\n";
    stbi_write_jpg(filename.c_str(), width, height, channelCount, dataCopy.get(), width * channelCount);
}

//bool copyRGBToCharArray(unsigned char *dest, const Color *src, int n) {
//
//    for (int i = 0; i < n; ++i) {
//        Color pixel = src[i];
//        pixel = clamp(gammaCorrection(pixel), 0.0, 0.999);
//        rgb[i] = {pixel.x(), pixel.y(), pixel.z()};
//    }
//}
