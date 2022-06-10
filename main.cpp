#include <iostream>
#include <vector>
#include <utils.h>
#include <Renderer.h>
#include "Ray.hpp"

int compare(int a) {
    return a == 15;
}

int main() {
    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    Renderer renderer(image_width, image_height);
    std::vector<Color> renderedImage = renderer.start();

    std::string filename = "test.jpg";
    int channelCount = 3;
    saveImgAsJpg(filename, renderedImage, image_width, image_height, channelCount);
    std::cout << "Done." << "\n";
    return 0;
}
