//
// Created by Sahar on 22/07/2022.
//

#include "ImGuiLayerUtils.h"

#include <imgui-docking/include/imgui.h>
#include <imgui-docking/include/imgui_impl_glfw.h>
#include <imgui-docking/include/imgui_impl_opengl3.h>


void ImGuiLayerUtils::startImGuiFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayerUtils::endImGuiFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void ImGuiLayerUtils::imGuiCleanup() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}