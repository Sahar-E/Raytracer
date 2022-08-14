//
// Created by Sahar on 22/07/2022.
//

#include "LayerHUD.cuh"
#include "utils.h"

#include <imgui-docking/include/imgui.h>
#include <imgui-docking/include/imgui_impl_glfw.h>
#include <imgui-docking/include/imgui_impl_opengl3.h>


LayerHUD::LayerHUD(GLFWwindow *window,
                   std::shared_ptr<LayerRGBStream> layerRGB,
                   std::string glslVersion) : _window(window), _layerRGB(std::move(layerRGB)),
                                              _glsl_version(std::move(glslVersion)) {
}

void LayerHUD::onUpdate() {
    ImGui::Begin("HUD Information", nullptr, ImGuiWindowFlags_::ImGuiWindowFlags_AlwaysAutoResize);

    imGuiFpsInfo();
    imGuiCameraSettings();
    imGuiRayTracerSettings();
    imGuiSaveImageSection();

    ImGui::End();
}

void LayerHUD::imGuiRayTracerSettings() {
    if (ImGui::CollapsingHeader("Ray Tracer settings:")) {
        imGuiNRenderCallsSlider();
        imGuiRenderWidthButtons();
        imGuiNRayBouncesSlider();
    }
}

void LayerHUD::imGuiNRayBouncesSlider() const {
    static int nRayBounces = _layerRGB->getRayTracerRenderer()->getNRayBounces();
    ImGui::SliderInt("# Ray Bounces", &nRayBounces, 1, 12);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        ImGui::SetTooltip("Number of RayTracer's Bounces for each ray cast into the scene..");
    }
    if (nRayBounces != _layerRGB->getRayTracerRenderer()->getNRayBounces()) {
        _layerRGB->getRayTracerRenderer()->setNRayBounces(nRayBounces);
    }
}

void LayerHUD::imGuiRenderWidthButtons() {
    ImGui::Text("Render width:");
    static int renderWidth = _layerRGB->getRendererImageWidth();
    imGuiRenderWidthButton(renderWidth, "80", 80);
    imGuiSameLineSpace();
    imGuiRenderWidthButton(renderWidth, "240", 240);
    imGuiSameLineSpace();
    imGuiRenderWidthButton(renderWidth, "400", 400);
    imGuiSameLineSpace();
    imGuiRenderWidthButton(renderWidth, "800", 800);

    imGuiRenderWidthButton(renderWidth, "1200", 1200);
    imGuiSameLineSpace();
    imGuiRenderWidthButton(renderWidth, "1600", 1600);
    imGuiSameLineSpace();
    imGuiRenderWidthButton(renderWidth, "2000", 2000);

    if (renderWidth != _layerRGB->getRendererImageWidth()) {
        _layerRGB->setRendererImageWidth(renderWidth);
    }
}

void LayerHUD::imGuiNRenderCallsSlider() {
    _nRenders = _layerRGB->getRendersPerFrame();
    ImGui::SliderInt("# RayTracer Render Calls per Frame", &_nRenders, 1, 15);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        ImGui::SetTooltip("Number of RayTracer Render Calls that will be made for each loop iteration in the main application GUI loop.");
    }
    if (_nRenders != _layerRGB->getRendersPerFrame()) {
        _layerRGB->setRendersPerFrame(_nRenders);
    }
}

void LayerHUD::imGuiRenderWidthButton(int & renderWidth, const char *buttonLabel, int afterPressWidth) const {
    if (ImGui::Button(buttonLabel)) {
        renderWidth = afterPressWidth;
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Width of Image that the ray tracer will render.");
    }
}

void LayerHUD::imGuiSameLineSpace() const {
    ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
}

void LayerHUD::imGuiCameraSettings() {
    if (ImGui::CollapsingHeader("Camera settings:", ImGuiTreeNodeFlags_FramePadding)) {
        imGuiCameraVFovSlider();
        imGuiCameraFocusDistSlider();
        imGuiCameraApertureSlider();
    }
}

void LayerHUD::imGuiCameraVFovSlider() {
    static float cameraVFov = _layerRGB->getCameraVFov();
    ImGui::SliderFloat("Camera Vertical Fov", &cameraVFov, 1.0f, 170.f);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Camera Vertical Fov measured in degrees.");
    }
    if (fabs(cameraVFov - _layerRGB->getCameraVFov()) > 0.00001f) {
        _layerRGB->setCameraVFov(cameraVFov);
    }
}

void LayerHUD::imGuiCameraFocusDistSlider() {
    static float cameraFocusDist = _layerRGB->getCameraFocusDist();
    ImGui::SliderFloat("Camera Focus Distance", &cameraFocusDist, 0.01f, 50.f);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Distance from the \"eye\" of the camera towards the scene.");
    }
    if (fabs(cameraFocusDist - _layerRGB->getCameraFocusDist()) > 0.00001f) {
        _layerRGB->setCameraFocusDist(cameraFocusDist);
    }
}

void LayerHUD::imGuiCameraApertureSlider() {
    static float cameraAperture = _layerRGB->getCameraAperture();
    ImGui::SliderFloat("Camera Aperture", &cameraAperture, 0.00001f, 0.1f);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("The smaller the aperture, the sharper the image.");
    }
    if (fabs(cameraAperture - _layerRGB->getCameraAperture()) > 0.00001f) {
        _layerRGB->setCameraAperture(cameraAperture);
    }
}

void LayerHUD::imGuiFpsInfo() {
    ImGui::Text("Application - average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                ImGui::GetIO().Framerate);
    _nRenders = _layerRGB->getRendersPerFrame();
    ImGui::Text("RayTracer   - average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate / _nRenders,
                ImGui::GetIO().Framerate * _nRenders);
}

void LayerHUD::imGuiSaveImageSection() const {
    if (ImGui::CollapsingHeader("Save:")) {
        static const int bufSize = 64;
        static char filename[64];
        static bool isFirstInit = false;
        if (isFirstInit) {
            strncpy(filename, "test.jpg", bufSize);
            isFirstInit = true;
        }
        ImGui::InputText("##", filename, bufSize);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Name of the file to save. Will be saved in the root directory of the executable.");
        }

        imGuiSameLineSpace();

        if (ImGui::Button("Save Image")) {
            saveImage(filename);
        }
    }
}

void LayerHUD::saveImage(const std::string &filename) const {
    auto rayTracer = _layerRGB->getRayTracerRenderer();
    saveImgAsJpg(filename, rayTracer->getPixelsOutAsChars(), rayTracer->getImgW(), rayTracer->getImgH());
}


void LayerHUD::onAttach() {
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(getWindow(), true);
    ImGui_ImplOpenGL3_Init(_glsl_version.c_str());
    ImGui::StyleColorsDark();
}

void LayerHUD::onDetach() {
}

GLFWwindow *LayerHUD::getWindow() const {
    return _window;
}

