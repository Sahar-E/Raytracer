//
// Created by Sahar on 17/07/2022.
//

#pragma once

#include <memory>
#include "RayTracerRenderer.cuh"
#include "glew-2.1.0/include/GL/glew.h"
#include "GLFW/glfw3.h"
#include "glm/detail/type_vec3.hpp"


class Application {
public:
    Application();

    int start();

private:
    int getGLWindow(GLFWwindow *&window, const float aspectRatio, const char *&glsl_version) const;

    void imguiCleanup() const;

    void imguiInit(GLFWwindow *window, const char *glsl_version) const;

    void startImguiFrame() const;

    void initGlBlendingConfigurations() const;
};
