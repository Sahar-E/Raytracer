#include <iostream>
#include <vector>
#include <Vec3.h>
#include <World.h>
#include <Camera.h>
#include <Renderer.h>
#include <utils.h>
#include <chrono>

World initWorld() {
    auto world = World();
    Color bluish = {0.2, 0.2, 1};
    Color red = {0.8, 0.2, 0.1};

    std::shared_ptr<Material> lambertianBlue = std::make_shared<Lambertian>(bluish);
    std::shared_ptr<Material> lambertianRed = std::make_shared<Lambertian>(red);

    world.addSphere(Sphere({0, 0, -2}, 0.5, lambertianBlue));
    world.addSphere(Sphere({0, -100.5, -2}, 100, lambertianRed));
    return world;
}

int main() {
    auto start = std::chrono::steady_clock::now();
    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int rayBounces = 12;
    int nSamplesPerPixel = 50;

    auto world = initWorld();
    auto camera = Camera({0, 0, 0}, aspect_ratio);
    Renderer renderer(image_width, image_height, world, camera, rayBounces, nSamplesPerPixel);
    std::vector<Color> renderedImage = renderer.render();

    std::string filename = "test.jpg";
    int channelCount = 3;
    saveImgAsJpg(filename, renderedImage, image_width, image_height, channelCount);

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Done." << "\n";
    std::cout << "Duration (ms): " << duration << "\n";
    return 0;
}
