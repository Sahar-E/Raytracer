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
    ImGui::Begin("HUD Information", nullptr, ImGuiWindowFlags_::ImGuiWindowFlags_AlwaysAutoResize);

    imGuiFpsInfo();
    imGuiCameraSettings();
    imGuiSaveImageFeature();

    ImGui::End();
}

void LayerHUD::imGuiCameraSettings() {
    ImGui::Text("Camera settings:");

    static float cameraVFov = _layerRGB->getCameraVFov();
    ImGui::SliderFloat("Camera Vertical Fov", &cameraVFov, 1.0f, 359.f);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        ImGui::SetTooltip("Camera Vertical Fov measured in degrees.");
    }
    if (fabs(cameraVFov - _layerRGB->getCameraVFov()) > 0.00001f) {
        _layerRGB->setCameraVFov(cameraVFov);
    }

    static float cameraFocusDist = _layerRGB->getCameraFocusDist();
    ImGui::SliderFloat("Camera Focus Distance", &cameraFocusDist, 0.01f, 50.f);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        ImGui::SetTooltip("Distance from the \"eye\" of the camera towards the scene.");
    }
    if (fabs(cameraFocusDist - _layerRGB->getCameraFocusDist()) > 0.00001f) {
        _layerRGB->setCameraFocusDist(cameraFocusDist);
    }

    static float cameraAperture = _layerRGB->getCameraAperture();
    ImGui::SliderFloat("Camera Aperture", &cameraAperture, 0.00001f, 0.1f);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        ImGui::SetTooltip("The smaller the aperture, the sharper the image.");
    }
    if (fabs(cameraAperture - _layerRGB->getCameraAperture()) > 0.00001f) {
        _layerRGB->setCameraAperture(cameraAperture);
    }
}

void LayerHUD::imGuiFpsInfo() {
    ImGui::Text("Application - average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                ImGui::GetIO().Framerate);

    static int nRenders = _layerRGB->getRendersPerFrame();
    ImGui::Text("RayTracer   - average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate / nRenders,
                ImGui::GetIO().Framerate * nRenders);
    ImGui::SliderInt("# RayTracer Render Calls per Frame", &nRenders, 1, 15);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        ImGui::SetTooltip("Number of RayTracer Render Calls that will be made for each loop iteration in the main application GUI loop.");
    }
    if (nRenders != _layerRGB->getRendersPerFrame()) {
        _layerRGB->setRendersPerFrame(nRenders);
    }
}

void LayerHUD::imGuiSaveImageFeature() const {
    ImGui::Text("Actions:");

    int bufSize = 32;
    static char filename[32];
    strncpy_s(filename, "test.jpg", bufSize);
    ImGui::InputText("##", filename, bufSize);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        ImGui::SetTooltip("Name of the file to save. Will be saved in the root directory of the executable.");
    }

    ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);

    if (ImGui::Button("Save Image")) {
        saveImage(filename);
    }
}

void LayerHUD::saveImage(const std::string &filename) const {
    const RayTracerRenderer &rayTracer = _layerRGB->getRayTracerRenderer();
    saveImgAsJpg(filename, rayTracer.getPixelsOutAsChars(), rayTracer.getImgW(), rayTracer.getImgH());
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

