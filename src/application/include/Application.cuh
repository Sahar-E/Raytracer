//
// Created by Sahar on 17/07/2022.
//

#pragma once

#include <memory>
#include "RayTracerRenderer.cuh"
#include "glew-2.1.0/include/GL/glew.h"
#include "GLFW/glfw3.h"
#include "Window.h"
#include "Layer.cuh"
#include "MouseEvents.hpp"

struct Configurations {
    float aspectRatio;
    int image_width;
    int rayBounces;
    float vFov;
    float aperture;
    int windowWidth;
};

class Application {
public:
    static Application &getInstance();
    int start(const Configurations &configurations);
    void onEvent(Event &event);
    [[nodiscard]] const std::shared_ptr<Window> &getWindow() const;

private:

    virtual ~Application() = default;
    Application(const Application & application ) = default;
    Application &operator=(const Application &instance) = default;
    Application() = default;

public:

private:
    Configurations _config{};
    std::shared_ptr<Window> _window;
    std::vector<std::shared_ptr<Layer>> _layers;

    void attachLayers();
};
