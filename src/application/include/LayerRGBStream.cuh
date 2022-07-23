//
// Created by Sahar on 22/07/2022.
//

#pragma once


#include "glew-2.1.0/include/GL/glew.h"
#include "GLFW/glfw3.h"
#include "Layer.cuh"
#include <memory>

#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"
#include "LiveTexture.h"
#include "InputHandler.h"
#include "VertexDrawer.h"
#include "RayTracerRenderer.cuh"
#include "Application.cuh"


static const float CAMERA_TRANSLATION_SIZE = .1f;
static const float CAMERA_ROT_SPEED = 0.08f;

class LayerRGBStream : public Layer {
public:
    LayerRGBStream(std::shared_ptr<Window> window, const Configurations &configurations);

    void onUpdate() override;
    void onAttach() override;

    const RayTracerRenderer &getRayTracerRenderer() const;

    int getRendersPerFrame() const;
    void setRendersPerFrame(int rendersPerFrame);

    void setCameraVFov(float cameraVFov);
    void setCameraAperture(float cameraAperture);
    void setCameraFocusDist(float cameraFocusDist);
    float getCameraVFov();
    float getCameraAperture();
    float getCameraFocusDist();

private:
    void updateCameraMovements();
    bool updateCameraRotations();
    bool updateCameraTranslations();
    static void mouseButtonCallback_releaseReturnCursorToNormal(GLFWwindow *window, int button, int action, int mods);
    void initCamera(const Configurations &config);
    void initRayTracer(const Configurations &config, const World &world);
    void setMouseButtonReleaseReturnCursorToNormal() const;
    World initWorld(const Configurations &config) const;
private:
    std::shared_ptr<Camera> _camera;
    std::shared_ptr<RayTracerRenderer> _rayTracerRenderer;

    std::shared_ptr<VertexArray> _va;
    std::shared_ptr<VertexBuffer> _vb;
    std::shared_ptr<VertexBufferLayout> _layout;
    std::shared_ptr<IndexBuffer> _ib;
    std::shared_ptr<Shader> _shader;
    std::shared_ptr<LiveTexture> _texture;

    glm::mat4 _proj;
    glm::mat4 _view;
    glm::mat4 _model;

    std::shared_ptr<InputHandler> _inputHandler;
    std::shared_ptr<Window> _window;
    int _rendersPerFrame{4};

};
