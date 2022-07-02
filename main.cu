#include <iostream>
#include <vector>
#include <World.cuh>
#include <Camera.cuh>
#include <Renderer.cuh>
#include "utils.cuh"
#include "TimeThis.h"
#include "commonDefines.h"
#include <string>
#include <cassert>


int main() {
    TimeThis t("main");
    const auto aspectRatio = 3.0 / 2.0;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspectRatio);
    const int rayBounces = 7;
    int vFov = 26;
    double aperture = 0.05;
    double focusDist = 10.0;
    int nFrames = 500;

    assert(0 < rayBounces && rayBounces <= MAX_BOUNCES);

    Vec3 vUp = {0, 1, 0};
    Vec3 lookFrom = {0, 1.8, 12};
    Vec3 lookAt = {0., 0, 0};



    auto world = World::initWorld2();
//    std::cout << "Size: " << world.getNSpheres()  << "\n";
    auto camera = Camera(lookFrom, lookAt, vUp, aspectRatio, vFov, aperture, focusDist);
    Renderer renderer(image_width, image_height, world, camera, rayBounces);

    for (int j = 0; j < nFrames; ++j) {
        renderer.render();
        std::cout << "Done iteration #: " << j  << "\n";
    }

    std::string filename = "test.jpg";
    int channelCount = 3;
    std::vector<std::tuple<double, double, double>> rgb(renderer.getNPixelsOut(), {0, 0, 0});
    for (int i = 0; i < renderer.getNPixelsOut(); ++i) {
        Color pixel = renderer.getPixelsOut()[i];
        pixel = clamp(gammaCorrection(pixel), 0.0, 0.999);
        rgb[i] = {pixel.x(), pixel.y(), pixel.z()};
    }
    saveImgAsJpg(filename, rgb, image_width, image_height, channelCount);

    std::cout << "Done." << "\n";
    return 0;
}