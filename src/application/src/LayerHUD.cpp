//
// Created by Sahar on 22/07/2022.
//

#include "LayerHUD.h"

#include <imgui-docking/include/imgui.h>
#include <imgui-docking/include/imgui_impl_glfw.h>
#include <imgui-docking/include/imgui_impl_opengl3.h>


LayerHUD::LayerHUD(GLFWwindow *window, std::string glslVersion) : _window(window), _glsl_version(glslVersion) {
}

void LayerHUD::onUpdate() {
    ImGui::Begin("SmallWindow!");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                ImGui::GetIO().Framerate);
    ImGui::End();
}

void LayerHUD::imGuiInit(GLFWwindow *window, const char *glsl_version) {
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    ImGui::StyleColorsDark();
}

void LayerHUD::onAttach() {
    imGuiInit(_window, _glsl_version.c_str());
}

void LayerHUD::onDetach() {
}

