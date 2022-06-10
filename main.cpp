#include <iostream>
#include <vector>
#include <utils.h>
#include "Ray.hpp"
#include "src/ray_tracing_library/include/Renderer.h"
#include "src/ray_tracing_library/include/Camera.h"

int compare(int a) {
    return a == 15;
}

World initWorld() {
    auto world = World();
    world.addSphere(Sphere({0, 0, -2}, 0.5));
    return world;
}

int main() {
    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);


    auto world = initWorld();
    auto camera = Camera({0, 0, 0}, aspect_ratio);
    Renderer renderer(image_width, image_height, world, camera);
    std::vector<Color> renderedImage = renderer.render();

    std::string filename = "test.jpg";
    int channelCount = 3;
    saveImgAsJpg(filename, renderedImage, image_width, image_height, channelCount);
    std::cout << "Done." << "\n";
    return 0;
}
