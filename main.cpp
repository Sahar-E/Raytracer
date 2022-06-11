#include <iostream>
#include <vector>
#include <Vec3.h>
#include <World.h>
#include <Camera.h>
#include <Renderer.h>
#include <utils.h>

World initWorld() {
    auto world = World();
    world.addSphere(Sphere({0, 0, -2}, 0.5));
    world.addSphere(Sphere({0, -100.5, -2}, 100));
    return world;
}

int main() {
    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int rayBounces = 50;
    int nSamplesPerPixel = 100;

    auto world = initWorld();
    auto camera = Camera({0, 0, 0}, aspect_ratio);
    Renderer renderer(image_width, image_height, world, camera, rayBounces, nSamplesPerPixel);
    std::vector<Color> renderedImage = renderer.render();

    std::string filename = "test.jpg";
    Vec3 v = randomVec();
    std::cout << v;
    int channelCount = 3;
    saveImgAsJpg(filename, renderedImage, image_width, image_height, channelCount);
    std::cout << "Done." << "\n";
    return 0;
}
