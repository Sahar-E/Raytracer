//
// Created by Sahar on 22/07/2022.
//

#include <iostream>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "LayerRGBStream.cuh"
#include "commonOpenGL.h"
#include "Application.cuh"
#include "Vec3.cuh"
#include "commonDefines.h"
#include "Event.hpp"
#include <imgui-docking/include/imgui.h>



LayerRGBStream::LayerRGBStream(std::shared_ptr<Window> window, const Configurations &config)
        : _window(std::move(window)),
          _proj(glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f)),
          _view(glm::translate(glm::mat4(1.0f), glm::vec3(-0, 0, 0))),
          _model(glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, 1.0f))),
          _rendererAspectRatio(config.aspectRatio),
          _rendererImageWidth(config.image_width),
          _rayBounces(config.rayBounces){
    _world = std::make_shared<World>(initWorld(config));
    initCamera(config);

    initRayTracerRenderer();
}

void LayerRGBStream::initRayTracerRenderer() {
    int imageWidth = _rendererImageWidth;
    int imageHeight = static_cast<int>(_rendererImageWidth / _rendererAspectRatio);
    _rayTracerRenderer = std::make_shared<RayTracerRenderer>(imageWidth, imageHeight, *_world, _camera,
                                                             _rayBounces);
}

World LayerRGBStream::initWorld(const Configurations &config) const {
    assert(0 < config.rayBounces && config.rayBounces <= MAX_BOUNCES);

    auto world = World::initWorld1();
    assert(world.getTotalSizeInMemoryForObjects() < 48 * pow(2, 10) &&
           "There is a hard limit for NVIDIA's shared memory size of 48KB for one block.");
    return world;
}

void LayerRGBStream::initCamera(const Configurations &config) {
    Vec3 vUp = {0, 1, 0};

    // Arbitrary Default Camera configuration.
    Vec3 lookFrom = {0, 0.5, 2.5};
    Vec3 lookAt = {0., 0.2, -2};
    _camera = std::make_shared<Camera>(lookFrom, lookAt, vUp, config.aspectRatio,
                                       config.vFov,
                                       config.aperture,
                                       (lookFrom - lookAt).length());
}

void LayerRGBStream::onUpdate() {
    for (int i = 0; i < _rendersPerFrame; i++) {
        _rayTracerRenderer->render();
    }
    _rayTracerRenderer->syncPixelsOutAsChars();
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
    if (InputHandler::isKeyDown(GLFW_KEY_A)) {
        _camera->moveCameraRight(-CAMERA_TRANSLATION_SIZE);
        cameraChanged = true;
    }
    if (InputHandler::isKeyDown(GLFW_KEY_D)) {
        _camera->moveCameraRight(CAMERA_TRANSLATION_SIZE);
        cameraChanged = true;
    }
    if (InputHandler::isKeyDown(GLFW_KEY_W)) {
        _camera->moveCameraForward(CAMERA_TRANSLATION_SIZE);
        cameraChanged = true;
    }
    if (InputHandler::isKeyDown(GLFW_KEY_S)) {
        _camera->moveCameraForward(-CAMERA_TRANSLATION_SIZE);
        cameraChanged = true;
    }
    if (InputHandler::isKeyDown(GLFW_KEY_SPACE) &&
        !InputHandler::isKeyDown(GLFW_KEY_LEFT_SHIFT)) {
        _camera->moveCameraUp(CAMERA_TRANSLATION_SIZE);
        cameraChanged = true;
    }
    if (InputHandler::isKeyDown(GLFW_KEY_SPACE) &&
        InputHandler::isKeyDown(GLFW_KEY_LEFT_SHIFT)) {
        _camera->moveCameraUp(-CAMERA_TRANSLATION_SIZE);
        cameraChanged = true;
    }
    return cameraChanged;
}


bool LayerRGBStream::updateCameraRotations() {
    bool cameraChanged = false;
    bool mouseNotOverImGui = !ImGui::GetIO().WantCaptureMouse;
    bool mouseButtonPressed = InputHandler::isMousePressed(GLFW_MOUSE_BUTTON_LEFT);
    float dX = ImGui::GetIO().MouseDelta.x * CAMERA_ROT_SPEED;
    float dY = ImGui::GetIO().MouseDelta.y * CAMERA_ROT_SPEED;
    bool isMouseMoved = dX != 0 || dY != 0;
    if (isMouseMoved && mouseNotOverImGui && mouseButtonPressed) {
        dX = ImGui::GetIO().MouseDelta.x * CAMERA_ROT_SPEED;
        dY = ImGui::GetIO().MouseDelta.y * CAMERA_ROT_SPEED;
        if (dX != 0.0f || dY != 0.0f) {
            _camera->rotateCamera(-dX, dY);
            cameraChanged = true;
        }
    }
    return cameraChanged;
}

void LayerRGBStream::onAttach() {
    initOpenGLBuffers();
}

void LayerRGBStream::initOpenGLBuffers() {
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

int LayerRGBStream::getRendersPerFrame() const {
    return _rendersPerFrame;
}

void LayerRGBStream::setRendersPerFrame(int rendersPerFrame) {
    _rendersPerFrame = rendersPerFrame;
}

void LayerRGBStream::setCameraVFov(float cameraVFov) {
    _camera->setVFov(cameraVFov);
    _rayTracerRenderer->clearPixels();
}

void LayerRGBStream::setCameraAperture(float cameraAperture) {
    _camera->setAperture(cameraAperture);
    _rayTracerRenderer->clearPixels();
}

void LayerRGBStream::setCameraFocusDist(float cameraFocusDist) {
    _camera->setFocusDist(cameraFocusDist);
    _rayTracerRenderer->clearPixels();
}

float LayerRGBStream::getCameraVFov() {
    return _camera->getVFov();
}

float LayerRGBStream::getCameraAperture() {
    return _camera->getAperture();
}

float LayerRGBStream::getCameraFocusDist() {
    return _camera->getFocusDist();
}

int LayerRGBStream::getRendererImageWidth() const {
    return _rendererImageWidth;
}

void LayerRGBStream::setRendererImageWidth(int rendererImageWidth) {
    _rendererImageWidth = rendererImageWidth;
    initRayTracerRenderer();
    initOpenGLBuffers();
}

const std::shared_ptr<RayTracerRenderer> &LayerRGBStream::getRayTracerRenderer() const {
    return _rayTracerRenderer;
}


void LayerRGBStream::onEvent(Event &event) {
    EventDispatcher dispatcher(event);

    dispatchMousePress(dispatcher);
    dispatchMouseRelease(dispatcher);
}

void LayerRGBStream::dispatchMouseRelease(EventDispatcher &dispatcher) const {
    dispatcher.dispatch<MouseButtonReleasedEvent>([this](MouseButtonReleasedEvent &event){
        GLFWwindow *window = _window->getWindow();
        int state = glfwGetInputMode(window, GLFW_CURSOR);
        if (state == GLFW_CURSOR_DISABLED) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        return true;
    });
}

void LayerRGBStream::dispatchMousePress(EventDispatcher &dispatcher) const {
    dispatcher.dispatch<MouseButtonPressedEvent>([this](MouseButtonPressedEvent &event){
        GLFWwindow *window = _window->getWindow();
        int state = glfwGetInputMode(window, GLFW_CURSOR);
        if (state == GLFW_CURSOR_NORMAL) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        return true;
    });
}

