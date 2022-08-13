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
#include "InputHandler.cuh"
#include "VertexDrawer.h"
#include "RayTracerRenderer.cuh"
#include "Application.cuh"
#include "MouseEvents.hpp"


static const float CAMERA_TRANSLATION_SIZE = .1f;
static const float CAMERA_ROT_SPEED = 0.08f;

/**
 * This class implements the rendering of the camera scene.
 */
class LayerRGBStream : public Layer {
public:

    /**
     * Construct a new LayerRGBStream with the given configurations.
     * @param window            The window containing the rendering.
     * @param configurations    The configurations of the rendering.
     */
    LayerRGBStream(std::shared_ptr<Window> window, const Configurations &configurations);

    void onUpdate() override;
    void onAttach() override;
    void onEvent(Event &event) override;

    const std::shared_ptr<RayTracerRenderer> &getRayTracerRenderer() const;

    int getRendersPerFrame() const;
    void setRendersPerFrame(int rendersPerFrame);

    void setCameraVFov(float cameraVFov);
    float getCameraVFov();

    void setCameraAperture(float cameraAperture);
    float getCameraAperture();

    void setCameraFocusDist(float cameraFocusDist);
    float getCameraFocusDist();

    int getRendererImageWidth() const;
    void setRendererImageWidth(int rendererImageWidth);

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
    int _rendersPerFrame{3};

    std::shared_ptr<World> _world;
    float _rendererAspectRatio;
    int _rendererImageWidth;
    int _rayBounces;

    /**
     * Init the RayTracerRenderer.
     */
    void initRayTracerRenderer();

    /**
     * Init the Camera.
     * @param config    Camera initial configurations.
     */
    void initCamera(const Configurations &config);

    /**
     * Init the buffers of OpenGL to be used to show the rendering of the RayTracerRenderer.
     */
    void initOpenGLBuffers();

    /**
     * Dispatch a mouse press event that will make the mouse cursor disabled only when pressed.
     * @param dispatcher    The EventDispatcher.
     */
    void dispatchMousePressEvent(EventDispatcher &dispatcher) const;

    /**
     * Dispatch a mouse release event that will make the mouse cursor enabled.
     * @param dispatcher    The EventDispatcher.
     */
    void dispatchMouseRelease(EventDispatcher &dispatcher) const;

    /**
     * Dispatch a window resize event that will make the RGB stream be compatible with the new aspect ratio.
     * @param dispatcher    The EventDispatcher.
     */
    void dispatchWindowResizeEvent(EventDispatcher &dispatcher);

    /**
     * Called to update the camera movement, translation and rotation.
     */
    void updateCameraMovements();

    /**
     * Updates the camera rotations.
     * @return  true if camera changed, false otherwise.
     */
    bool updateCameraRotations();

    /**
     * Called to update the camera position.
     * @return  true if camera changed, false otherwise.
     */
    bool updateCameraTranslations();


    /**
     * Init a world scene for the RayTracerRenderer.
     * @param config the world configurations.
     * @return  the new scene.
     */
    World initWorld(const Configurations &config) const;
};
