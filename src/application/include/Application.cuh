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

/**
 * The Raytracer application.
 */
class Application {
public:
    static Application &getInstance();

    Application(const Application &application) = delete;

    Application &operator=(const Application &instance) = delete;

    /**
     * Starts the Raytracer application.
     * @param configurations    The configuration to use.
     */
    void start(const Configurations &configurations);

    /**
     * Called when the application receive an event.
     * @param event     The event.
     */
    void onEvent(Event &event);

    [[nodiscard]] const std::shared_ptr<Window> &getWindow() const;

private:
    Configurations _config{};
    std::shared_ptr<Window> _window;
    std::vector<std::shared_ptr<Layer>> _layers;

    virtual ~Application() = default;

    Application() = default;

    void attachLayers();

    void onWindowResize(EventDispatcher &dispatcher);
};
