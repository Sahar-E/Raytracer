#include <iostream>
#include "Application.cuh"


int main() {
    float aspectRatio = 16.0f / 9.0f;
    int image_width = 1200;
    int image_height = static_cast<int>(image_width / aspectRatio);
    int rayBounces = 7;
    float vFov = 26.0f;
    float aperture = 0.005f;

    int windowWidth = 2000;

    Configurations configurations = {aspectRatio, image_width, image_height, rayBounces, vFov, aperture, windowWidth};
    Application &app = Application::getInstance();
    app.start(configurations);
    return 0;
}