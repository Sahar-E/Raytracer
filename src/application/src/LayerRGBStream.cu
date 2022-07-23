//
// Created by Sahar on 22/07/2022.
//

#include <iostream>
#include <utility>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "LayerRGBStream.cuh"
#include "commonOpenGL.h"
#include "Application.cuh"
#include "Vec3.cuh"
#include "commonDefines.h"
#include <imgui-docking/include/imgui.h>


void LayerRGBStream::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
    int state = glfwGetInputMode(window, GLFW_CURSOR);
    if (state == GLFW_CURSOR_DISABLED) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
}

LayerRGBStream::LayerRGBStream(std::shared_ptr<Window> window, const Configurations &config)
        : _window(std::move(window)),
          _proj(glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f)),
          _view(glm::translate(glm::mat4(1.0f), glm::vec3(-0, 0, 0))),
          _model(glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, 1.0f))) {
    assert(0 < config.rayBounces && config.rayBounces <= MAX_BOUNCES);

    Vec3 vUp = {0, 1, 0};
    Vec3 lookFrom = {0, 0.5, 2.5};
    Vec3 lookAt = {0., 0.2, -2};
    float focusDist = (lookFrom - lookAt).length();

    auto world = World::initWorld1();
    assert(world.getTotalSizeInMemoryForObjects() < 48 * pow(2, 10) &&
           "There is a hard limit for NVIDIA's shared memory size of 48KB for one block.");

    _camera = std::make_shared<Camera>(lookFrom, lookAt, vUp, config.aspectRatio, config.vFov, config.aperture,
                                       focusDist);
    _rayTracerRenderer = std::make_shared<RayTracerRenderer>(config.image_width, config.image_height, world, _camera,
                                                             config.rayBounces);
    glfwSetMouseButtonCallback(_window->getWindow(), mouseButtonCallback);
}

void LayerRGBStream::onUpdate() {
    _rayTracerRenderer->render();   // TODO-Sahar: move to different thread.
    _rayTracerRenderer->syncPixelsOut();
    updateCameraMovements();

    _texture->updateTexture();

    glm::mat4 mvp = _proj * _view * _model;
    _shader->bind();
    _shader->setUniformMat4f("u_mvpMatrix", mvp);
    VertexDrawer::draw(*_va, *_ib, *_shader);


}

void LayerRGBStream::updateCameraMovements() {
    bool cameraChanged = false;
    cameraChanged |= updateCameraRotations();
    cameraChanged |= updateCameraTranslations();
    if (cameraChanged) {
        _rayTracerRenderer->clearPixels();
    }
}

bool LayerRGBStream::updateCameraTranslations() {
    bool cameraChanged = false;
    if (_window->getInputHandler()->isKeyDown(GLFW_KEY_A)) {
        _camera->moveCameraRight(-CAMERA_TRANSLATION_SIZE);
        cameraChanged = true;
    }
    if (_window->getInputHandler()->isKeyDown(GLFW_KEY_D)) {
        _camera->moveCameraRight(CAMERA_TRANSLATION_SIZE);
        cameraChanged = true;
    }
    if (_window->getInputHandler()->isKeyDown(GLFW_KEY_W)) {
        _camera->moveCameraForward(CAMERA_TRANSLATION_SIZE);
        cameraChanged = true;
    }
    if (_window->getInputHandler()->isKeyDown(GLFW_KEY_S)) {
        _camera->moveCameraForward(-CAMERA_TRANSLATION_SIZE);
        cameraChanged = true;
    }
    if (_window->getInputHandler()->isKeyDown(GLFW_KEY_SPACE) &&
        !_window->getInputHandler()->isKeyDown(GLFW_KEY_LEFT_SHIFT)) {
        _camera->moveCameraUp(CAMERA_TRANSLATION_SIZE);
        cameraChanged = true;
    }
    if (_window->getInputHandler()->isKeyDown(GLFW_KEY_SPACE) &&
        _window->getInputHandler()->isKeyDown(GLFW_KEY_LEFT_SHIFT)) {
        _camera->moveCameraUp(-CAMERA_TRANSLATION_SIZE);
        cameraChanged = true;
    }
    return cameraChanged;
}


bool LayerRGBStream::updateCameraRotations() {
    bool cameraChanged = false;
    bool isMouseMove = _window->getInputHandler()->isMouseMove();
    bool mouseNotOverImGui = !ImGui::GetIO().WantCaptureMouse;
    bool mouseButtonPressed = glfwGetMouseButton(_window->getWindow(), GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (isMouseMove && mouseNotOverImGui && mouseButtonPressed) {
        glfwSetInputMode(_window->getWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        float dX = ImGui::GetIO().MouseDelta.x * CAMERA_ROT_SPEED;
        float dY = ImGui::GetIO().MouseDelta.y * CAMERA_ROT_SPEED;
        if (dX != 0.0f || dY != 0.0f) {
            _camera->rotateCamera(-dX, dY);
            cameraChanged = true;
        }
    }
    return cameraChanged;
}

void LayerRGBStream::onAttach() {
    float positions[] = {
            -0.9999f, -0.9999f, 0.0f, 0.0f,
            0.9999f, -0.9999f, 1.0f, 0.0f,
            0.9999f, 0.9999f, 1.0f, 1.0f,
            -0.9999f, 0.9999f, 0.0f, 1.0f
    };
    unsigned int indices[] = {
            0, 1, 2,
            2, 3, 0
    };


    _va = std::make_shared<VertexArray>();
    _vb = std::make_shared<VertexBuffer>(positions, sizeof(float) * 4 * 4);

    _layout = std::make_shared<VertexBufferLayout>();
    _layout->push<float>(2);
    _layout->push<float>(2);
    _va->addBuffer(*_vb, *_layout);

    _ib = std::make_shared<IndexBuffer>(indices, 6);

    _shader = std::make_shared<Shader>("resources/shaders/Basic.shader");
    _shader->bind();
    _texture = std::make_shared<LiveTexture>(_rayTracerRenderer->getPixelsOutAsChars(),
                                             _rayTracerRenderer->getImgW(),
                                             _rayTracerRenderer->getImgH(), GL_RGB);
    _texture->bind();
    _shader->setUniform1i("u_texture", 0);

    _va->unbind();
    _vb->unbind();
    _ib->unbind();
    _shader->unbind();
}

const std::shared_ptr<RayTracerRenderer> &LayerRGBStream::getRayTracerRenderer() const {
    return _rayTracerRenderer;
}

