//
// Created by Sahar on 17/07/2022.
//

#include "Application.cuh"
#include <string>
#include <memory>
#include "glfw-3.3.7/include/GLFW/glfw3.h"
#include "VertexDrawer.h"
#include "stb_library/include/stb_library/stb_image.h"
#include "InputHandler.h"
#include "Window.h"
#include "Layer.cuh"
#include "LayerRGBStream.cuh"
#include "LayerHUD.cuh"
#include "ImGuiLayerUtils.h"


int Application::start(const Configurations &configurations) {
    _config = configurations;

    _window = std::make_shared<Window>("RayTracer", _config.aspectRatio, _config.windowWidth);
    _layers = std::vector<std::shared_ptr<Layer>>();
    attachLayers();

    /* Loop until the user closes the window */
    while (!_window->shouldClose()) {
        ImGuiLayerUtils::startImGuiFrame();
        VertexDrawer::clear();

        for (const auto &layer: _layers) {
            layer->onUpdate();
        }

        ImGuiLayerUtils::endImGuiFrame();
        /* Swap front and back buffers */
        glfwSwapBuffers(_window->getWindow());
        /* Poll for and process events */
        glfwPollEvents();
    }

    ImGuiLayerUtils::imGuiCleanup();
    _layers.clear();
    glfwTerminate();
    return 0;
}

void Application::attachLayers() {
    std::shared_ptr<LayerRGBStream> layerRGB = std::make_shared<LayerRGBStream>(_window, _config);
    _layers.push_back(layerRGB);
    _layers.push_back(std::make_shared<LayerHUD>(_window->getWindow(), layerRGB, _window->getGlslVersion()));

    for (const auto &layer: _layers) {
        layer->onAttach();
    }
}


Application &Application::getInstance() {
    static Application INSTANCE;
    return INSTANCE;
}

const std::shared_ptr<Window> &Application::getWindow() const {
    return _window;
}



