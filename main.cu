#include <iostream>
#include "Application.cuh"


int main() {
    float aspectRatio = 3.0f / 2.0f;
    int image_width = 400;
    int image_height = static_cast<int>(image_width / aspectRatio);
    int rayBounces = 7;
    float vFov = 26.0f;
    float aperture = 0.005f;

    Configurations configurations = {aspectRatio, image_width, image_height, rayBounces, vFov, aperture};
    Application &app = Application::getInstance();
    app.start(configurations);
    return 0;
}


//int main() {
//    TimeThis timeThis;
//    const auto aspectRatio = 3.0f / 2.0f;
//    const int image_width = 1200;
//    const int image_height = static_cast<int>(image_width / aspectRatio);
//    const int rayBounces = 7;
//    float vFov = 26.0f;
//    float aperture = 0.005f;
//    int nFrames = 1;
//
//    assert(0 < rayBounces && rayBounces <= MAX_BOUNCES);
//
//    Vec3 vUp = {0, 1, 0};
//    Vec3 lookFrom = {0, 0.5, 2.5};
//    Vec3 lookAt = {0., 0.2, 0};
//    float focusDist = (lookFrom - lookAt).length();
//
//
//
//    auto world = World::initWorld1();
//    std::cout << "Size: " << world.getTotalSizeInMemoryForObjects() << "\n";
//    std::cout << "nSpheres: " << world.getNSpheres()  << "\n";
//    assert(world.getTotalSizeInMemoryForObjects() < 48 * pow(2, 10) && "There is a hard limit for NVIDIA's shared memory size of 48KB for one block.");
//    auto camera = Camera(lookFrom, lookAt, vUp, aspectRatio, vFov, aperture, focusDist);
//    Renderer renderer(image_width, image_height, world, camera, rayBounces);
//
//    for (int j = 0; j < nFrames; ++j) {
//        renderer.render();
//        std::cout << "Done iteration #: " << j  << "\n";
//    }
//
//    std::string filename = "test.jpg";
//    int channelCount = 3;
//    std::vector<std::tuple<float, float, float>> rgb(renderer.getNPixelsOut(), {0, 0, 0});
//    for (int i = 0; i < renderer.getNPixelsOut(); ++i) {
//        Color pixel = renderer.getPixelsOut()[i];
//        pixel = clamp(gammaCorrection(pixel), 0.0, 0.999);
//        rgb[i] = {pixel.x(), pixel.y(), pixel.z()};
//    }
//    saveImgAsJpg(filename, rgb, image_width, image_height, channelCount);
//
//    std::cout << "Done." << "\n";
//    return 0;
//}