//
// Created by Sahar on 17/07/2022.
//

#pragma once

#include <memory>
#include "RayTracerRenderer.cuh"
#include "glew-2.1.0/include/GL/glew.h"
#include "GLFW/glfw3.h"
#include "Window.h"
#include "Layer.h"


struct Configurations {
    float aspectRatio;
    int image_width;
    int image_height;
    int rayBounces;
    float vFov;
    float aperture;
    int windowWidth;
};

class Application {
public:
    static Application &getInstance();
    int start(const Configurations &configurations);



private:

    virtual ~Application() = default;
    Application(const Application & application ) = default;
    Application &operator=(const Application &instance) = default;
    Application() = default;

private:
    Configurations _config{};
    std::shared_ptr<Window> _window;
    std::shared_ptr<Camera> _camera;
    std::shared_ptr<RayTracerRenderer> _rayTracerRenderer;
    std::vector<std::shared_ptr<Layer>> _layers;
};
