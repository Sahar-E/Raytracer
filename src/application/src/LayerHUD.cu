//
// Created by Sahar on 22/07/2022.
//

#include "LayerHUD.cuh"
#include "utils.h"

#include <imgui-docking/include/imgui.h>
#include <imgui-docking/include/imgui_impl_glfw.h>
#include <imgui-docking/include/imgui_impl_opengl3.h>

#include <utility>


LayerHUD::LayerHUD(GLFWwindow *window,
                   std::shared_ptr<LayerRGBStream> layerRGB,
                   std::string glslVersion) : _window(window), _layerRGB(std::move(layerRGB)),
                                              _glsl_version(std::move(glslVersion)) {
}

void LayerHUD::onUpdate() {
    ImGui::Begin("HUD Information");

    textFPS();
    saveImageFeature();

    ImGui::End();
}

void LayerHUD::saveImageFeature() const {
    int bufSize = 32;
    static char filename[32];
    strncpy_s(filename, "test.jpg", bufSize);
    ImGui::InputText("", filename, bufSize);
    ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
    if (ImGui::Button("Save Image")) {
        saveImage(filename);
    }
}

void LayerHUD::saveImage(const std::string &filename) const {
    std::shared_ptr<RayTracerRenderer> rayTracer = _layerRGB->getRayTracerRenderer();
    saveImgAsJpg(filename, rayTracer->getPixelsOutAsChars(), rayTracer->getImgW(), rayTracer->getImgH());
}

void LayerHUD::textFPS() {
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                ImGui::GetIO().Framerate);
}

void LayerHUD::imGuiInit(GLFWwindow *window, const char *glsl_version) {
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    ImGui::StyleColorsDark();
}

void LayerHUD::onAttach() {
    imGuiInit(getWindow(), _glsl_version.c_str());
}

void LayerHUD::onDetach() {
}

GLFWwindow *LayerHUD::getWindow() const {
    return _window;
}

