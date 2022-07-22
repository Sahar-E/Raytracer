//
// Created by Sahar on 17/07/2022.
//

#pragma once

#include <memory>
#include "RayTracerRenderer.cuh"
#include "glew-2.1.0/include/GL/glew.h"
#include "GLFW/glfw3.h"


struct Configurations {
    float aspectRatio;
    int image_width;
    int image_height;
    int rayBounces;
    float vFov;
    float aperture;
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

    int getGLWindow(GLFWwindow *&window, const float aspectRatio, const char *&glsl_version) const;
    void imguiCleanup() const;
    void imguiInit(GLFWwindow *window, const char *glsl_version) const;
    void startImguiFrame() const;
    void initGlBlendingConfigurations() const;

public:
    [[nodiscard]] const std::shared_ptr<Camera> &getCamera() const;
    [[nodiscard]] const std::shared_ptr<RayTracerRenderer> &getRayTracerRenderer() const;

    [[nodiscard]] GLFWwindow *getWindow() const;

private:
    Configurations _config{};
    std::shared_ptr<Camera> _camera;
    std::shared_ptr<RayTracerRenderer> _rayTracerRenderer;
    GLFWwindow *_window;
};
