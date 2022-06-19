#include <iostream>
#include <vector>
#include <Vec3.h>
#include <World.h>
#include <Camera.h>
#include <Renderer.h>
#include <utils.h>
#include <chrono>

World initWorld1() {
    auto world = World();
    Color red = {0.8, 0.2, 0.1};
    Color white = {1,1,1};
    Color bezh = {227 / 255.0,203 / 255.0,165 / 255.0};
    Color green = {51 / 255.0, 83 / 255.0,69 / 255.0};
    Color brown = {118 / 255.0, 91 / 255.0,70 / 255.0};
    Color redBrown = {141 / 255.0, 78 / 255.0,44 / 255.0};
    Color darkGreen = {74 / 255.0, 71 / 255.0,51 / 255.0};

//    Material metalGreen = Material(whiteGreenish, 0.99);
    Material mirror = Material(white, 0.0);
    Material mirrorRedBrown = Material(redBrown, 0.0);
    Material lambertianBezh = Material::getLambertian(bezh);
    Material lambertianGreen = Material::getLambertian(green);
    Material lambertianBrown = Material::getLambertian(brown);
    Material lambertianRedBrown = Material::getLambertian(redBrown);
    Material metalGreen = Material(green, 0.0);
    Material lambertianDarkGreen = Material::getLambertian(darkGreen);

    world.addSphere(Sphere({0, 0, -2}, 0.5, lambertianRedBrown));
    world.addSphere(Sphere({1.2, 0, -3}, 0.5, mirror));
    world.addSphere(Sphere({-1.2, -0.2, -2.5}, 0.3, mirrorRedBrown));
    world.addSphere(Sphere({0.6, -0.4, -2.2}, 0.1, lambertianDarkGreen));
    world.addSphere(Sphere({0.4, -0.4, -1.2}, 0.1, lambertianBezh));
    world.addSphere(Sphere({1, -0.4, -1.5}, 0.1, metalGreen));
    world.addSphere(Sphere({-0.9, -0.4, -3.3}, 0.1, lambertianGreen));
    world.addSphere(Sphere({-0.5, -0.4, -1.6}, 0.1, mirror));
    world.addSphere(Sphere({-0.15, -0.4, -1.1}, 0.1, lambertianGreen));
    world.addSphere(Sphere({-0.8, -0.4, -1.3}, 0.1, lambertianRedBrown));
    world.addSphere(Sphere({0, -100.5, -2}, 100, lambertianBrown));
    return world;
}

World initWorld2() {
    auto world = World();
    Color bluish = {0.2, 0.2, 1};
    Color whiteGreenish = {0.4, 1, .4};
    Color red = {0.8, 0.2, 0.1};
    Color green = {0.2, 0.8, 0.1};
    Color white = {1,1,1};
    Color bezh = {0.96,0.96,0.86};

    Material lambertianBlue = Material::getLambertian(bluish);
    Material metalGreen = Material(whiteGreenish, 0.99);
    Material mirror = Material(white, 0.0);
    Material lambertianRed = Material::getLambertian(red);
    Material lambertianGreen = Material::getLambertian(green);
    Material lambertianBezh = Material::getLambertian(bezh);
//    Material glass = std::make_shared<Glass>(white, 1.5);

//    Material light = std::make_shared<Light>(white, 100);

    world.addSphere(Sphere({0, 0, -2}, 0.5, metalGreen));
    world.addSphere(Sphere({1, 0, -3}, 0.5, mirror));
    world.addSphere(Sphere({-1, 0, -2.5}, 0.5, lambertianBlue));
    world.addSphere(Sphere({-0.5, -0.4, -1.5}, 0.1, mirror));
//    world.addSphere(Sphere({0, 1, -2}, 0.2, light));                    // Light source

    world.addSphere(Sphere({0, -1000.5, -2}, 1000, lambertianBezh));    // Floor
//    world.addSphere(Sphere({-1000.5, 0, -2}, 1000, lambertianRed));     // left wall
//    world.addSphere(Sphere({1000.5, 0, -2}, 1000, lambertianGreen));    // left wall
    return world;
}

int main() {
    auto start = std::chrono::steady_clock::now();
    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 2400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int rayBounces = 12;
    int nSamplesPerPixel = 200;

    auto world = initWorld1();
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
