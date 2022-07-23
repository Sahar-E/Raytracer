//
// Created by Sahar on 22/07/2022.
//

#pragma once


#include "glew-2.1.0/include/GL/glew.h"
#include "GLFW/glfw3.h"
#include "Layer.cuh"
#include "LayerRGBStream.cuh"
#include <memory>

class LayerHUD : public Layer {
public:
    LayerHUD(GLFWwindow *window, std::shared_ptr<LayerRGBStream> layerRGB, std::string glslVersion);

    void onUpdate() override;

    void onAttach() override;

    void onDetach() override;

    GLFWwindow *getWindow() const;

private:
    static void imGuiInit(GLFWwindow *window, const char *glsl_version);

private:
    GLFWwindow *_window;
    std::shared_ptr<LayerRGBStream> _layerRGB;
    std::string _glsl_version;

    static void textFPS() ;

    void saveImage(const std::string &filename) const;

    void saveImageFeature() const;
};
