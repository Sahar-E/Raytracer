//
// Created by Sahar on 17/07/2022.
//

#include "Application.cuh"
#include <string>
#include <memory>
#include "glew-2.1.0/include/GL/glew.h"
#include "glfw-3.3.7/include/GLFW/glfw3.h"
#include "VertexDrawer.h"
#include <iostream>
#include "commonOpenGL.h"
#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"
#include "stb_library/include/stb_library/stb_image.h"
#include "LiveTexture.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "Vec3.cuh"
#include "World.cuh"
#include "Camera.cuh"
#include "RayTracerRenderer.cuh"
#include "InputHandler.h"
#include "Window.h"
#include "Layer.h"
#include "LayerRGBStream.h"
#include "commonDefines.h"
#include "LayerHUD.h"
#include "ImGuiLayerUtils.h"
#include <imgui-docking/include/imgui.h>
#include <imgui-docking/include/imgui_impl_glfw.h>
#include <imgui-docking/include/imgui_impl_opengl3.h>


int Application::start(const Configurations &configurations) {
    _config = configurations;
    assert(0 < _config.rayBounces && _config.rayBounces <= MAX_BOUNCES);

    Vec3 vUp = {0, 1, 0};
    Vec3 lookFrom = {0, 0.5, 2.5};
    Vec3 lookAt = {0., 0.2, -2};
    float focusDist = (lookFrom - lookAt).length();


    auto world = World::initWorld1();
    std::cout << "Size: " << world.getTotalSizeInMemoryForObjects() << "\n";
    std::cout << "nSpheres: " << world.getNSpheres()  << "\n";
    assert(world.getTotalSizeInMemoryForObjects() < 48 * pow(2, 10) && "There is a hard limit for NVIDIA's shared memory size of 48KB for one block.");
    _camera = std::make_shared<Camera>(lookFrom, lookAt, vUp, _config.aspectRatio, _config.vFov, _config.aperture, focusDist);
    _rayTracerRenderer = std::make_shared<RayTracerRenderer>(_config.image_width, _config.image_height, world, _camera, _config.rayBounces);


    _window = std::make_shared<Window>("RayTracer", _config.aspectRatio, _config.windowWidth);
    _layers = std::vector<std::shared_ptr<Layer>>();
    _layers.push_back(std::make_shared<LayerRGBStream>(_rayTracerRenderer->getPixelsOutAsChars(), _rayTracerRenderer->getImgW(), _rayTracerRenderer->getImgH()));
    _layers.push_back(std::make_shared<LayerHUD>(_window->getWindow(), _window->getGlslVersion()));

    {
        for (const auto &layer: _layers) {
            layer->onAttach();
        }


        /* Loop until the user closes the window */
        while (!_window->shouldClose()) {
            ImGuiLayerUtils::startImGuiFrame();
            VertexDrawer::clear();

            {
//                TimeThis t("Update texture");
                _rayTracerRenderer->render();
                _rayTracerRenderer->syncPixelsOut();
            }

            for (const auto &layer: _layers) {
                layer->onUpdate();
            }

            {
                bool cameraChanged = false;
                if (_window->getInputHandler()->isMouseMove() && !ImGui::GetIO().WantCaptureMouse) {
                    glfwSetInputMode(_window->getWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                    float speedFactor = 0.08f;
                    float dX = ImGui::GetIO().MouseDelta.x * speedFactor;
                    float dY = ImGui::GetIO().MouseDelta.y * speedFactor;
                    if (dX != 0.0f || dY != 0.0f) {
                        _camera->rotateCamera(-dX, dY);
                        cameraChanged = true;
                    }
                } else {
                    int state = glfwGetInputMode(_window->getWindow(), GLFW_CURSOR);
                    if (state == GLFW_CURSOR_DISABLED) {
                        glfwSetInputMode(_window->getWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                    }
                }

                if (_window->getInputHandler()->isKeyDown(GLFW_KEY_A)) {
                    _camera->moveCameraRight(-.1f);
                    cameraChanged = true;
                }
                if (_window->getInputHandler()->isKeyDown(GLFW_KEY_D)) {
                    _camera->moveCameraRight(.1f);
                    cameraChanged = true;
                }
                if (_window->getInputHandler()->isKeyDown(GLFW_KEY_W)) {
                    _camera->moveCameraForward(.1f);
                    cameraChanged = true;
                }
                if (_window->getInputHandler()->isKeyDown(GLFW_KEY_S)) {
                    _camera->moveCameraForward(-.1f);
                    cameraChanged = true;
                }
                if (_window->getInputHandler()->isKeyDown(GLFW_KEY_SPACE) && !_window->getInputHandler()->isKeyDown(GLFW_KEY_LEFT_SHIFT)) {
                    _camera->moveCameraUp(.1f);
                    cameraChanged = true;
                }
                if (_window->getInputHandler()->isKeyDown(GLFW_KEY_SPACE) && _window->getInputHandler()->isKeyDown(GLFW_KEY_LEFT_SHIFT)) {
                    _camera->moveCameraUp(-.1f);
                    cameraChanged = true;
                }

                if (cameraChanged) {
                    _rayTracerRenderer->clearPixels();
                }
            }


            ImGuiLayerUtils::endImGuiFrame();
            /* Swap front and back buffers */
            glfwSwapBuffers(_window->getWindow());
            /* Poll for and process events */
            glfwPollEvents();
        }
        ImGuiLayerUtils::imGuiCleanup();
    }

    glfwTerminate();
    return 0;
}


Application &Application::getInstance() {
    static Application INSTANCE;
    return INSTANCE;
}


