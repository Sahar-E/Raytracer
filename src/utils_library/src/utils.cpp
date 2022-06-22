//
// Created by Sahar on 08/06/2022.
//

#include "utils.h"
#include <string>
#include <vector>
#include <memory>
#include "constants.h"
#include "stb_library/stb_image_write.h"
#include "Vec3.h"

void saveImgAsJpg(const std::string &filename,
                  const std::vector<Vec3> &data,
                  const int width,
                  const int height,
                  const int channelCount) {
    int n = height * width;
    auto dataCopy = std::make_unique<unsigned char[]>(n * channelCount);
    for (int i = 0; i < n; ++i) {
        dataCopy[i * channelCount] = static_cast<unsigned char>(data[i].x() * 255);
        dataCopy[i * channelCount + 1] = static_cast<unsigned char>(data[i].y() * 255);
        dataCopy[i * channelCount + 2] = static_cast<unsigned char>(data[i].z() * 255);
    }
    std::cout << "\nSaving image to " << filename << "\n";
    stbi_write_jpg(filename.c_str(), width, height, channelCount, dataCopy.get(), width * channelCount);
}

Vec3 alphaBlending(Vec3 v1, Vec3 v2, double alpha) {
    return v1 * (1 - alpha) + v2 * alpha;
}

bool fcmp(double a, double b) {
    return std::fabs(a - b) < EPS;
}

Color gammaCorrection(Color const &color) {
    return {std::sqrt(color.x()), std::sqrt(color.y()), std::sqrt(color.z())};
}

double clamp(double toClamp, double low, double high) {
    if (toClamp < low) {
        return low;
    }
    if (toClamp > high) {
        return high;
    }
    return toClamp;
}

Vec3 clamp(const Vec3 &toClamp, double low, double high) {
    return {clamp(toClamp.x(), low, high),
            clamp(toClamp.y(), low, high),
            clamp(toClamp.z(), low, high)};
}

bool cannotRefractBySnellsLaw(double cosTheta, double refractionIdxRatio) {
    double sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    bool cannotRefract = refractionIdxRatio * sinTheta > 1.0;
    return cannotRefract;
}

double reflectSchlickApproxForFrensel(double cosTheta, double refractionIdxRatio) {
    double r0 = (1 - refractionIdxRatio) / (1 + refractionIdxRatio);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow(1 - cosTheta, 5);
}