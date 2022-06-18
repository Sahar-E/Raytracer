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
    Color whiteGreenish = {0.7, 1, .7};
    Color red = {0.8, 0.2, 0.1};
    Color white = {1,1,1};

    std::shared_ptr<Material> lambertianBlue = std::make_shared<Lambertian>(bluish);
    std::shared_ptr<Material> metalGreen = std::make_shared<Metal>(whiteGreenish, 0.0);
    std::shared_ptr<Material> mirror = std::make_shared<Metal>(white, 0.0);
    std::shared_ptr<Material> lambertianRed = std::make_shared<Lambertian>(red);

    world.addSphere(Sphere({0, 0, -2}, 0.5, mirror));
    world.addSphere(Sphere({1, 0, -3}, 0.5, mirror));
    world.addSphere(Sphere({-1, 0, -2.5}, 0.5, lambertianBlue));
    world.addSphere(Sphere({-0.5, -0.4, -1.5}, 0.1, mirror));
    world.addSphere(Sphere({0.3, -0.4, -1.2}, 0.1, lambertianBlue));
    world.addSphere(Sphere({0, -100.5, -2}, 100, lambertianRed));
    return world;
}

int main() {
    auto start = std::chrono::steady_clock::now();
    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 2400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int rayBounces = 12;
    int nSamplesPerPixel = 100;

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
