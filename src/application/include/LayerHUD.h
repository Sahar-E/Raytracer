//
// Created by Sahar on 22/07/2022.
//

#pragma once


#include "glew-2.1.0/include/GL/glew.h"
#include "GLFW/glfw3.h"
#include "Layer.h"

class LayerHUD : public Layer {
public:
    LayerHUD(GLFWwindow *window, std::string glslVersion);

    void onUpdate() override;

    void onAttach() override;

    void onDetach() override;

private:
    static void imGuiInit(GLFWwindow *window, const char *glsl_version);

private:
    GLFWwindow *_window;
    std::string _glsl_version;
};
