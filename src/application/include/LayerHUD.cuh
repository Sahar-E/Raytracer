//
// Created by Sahar on 22/07/2022.
//

#pragma once


#include "glew-2.1.0/include/GL/glew.h"
#include "GLFW/glfw3.h"
#include "Layer.cuh"
#include "LayerRGBStream.cuh"
#include <memory>

/**
 * This layer is the ImGui window that is responsible for the settings that can be changed in the application while it
 * is running.
 */
class LayerHUD : public Layer {
public:

    /**
     * Constructs a new Layer that will create a "window" with a "Heads up display" to apply the settings in the
     * application.
     * @param window        The window to create the GUI inside.
     * @param layerRGB      The RGB layer to apply the settings to.
     * @param glslVersion   The version of the OpenGL.
     */
    LayerHUD(GLFWwindow *window, std::shared_ptr<LayerRGBStream> layerRGB, std::string glslVersion);

    void onUpdate() override;

    void onAttach() override;

    void onDetach() override;

    GLFWwindow *getWindow() const;


private:
    GLFWwindow *_window;
    std::shared_ptr<LayerRGBStream> _layerRGB;
    std::string _glsl_version;
    int _nRenders;

    /**
     * Save the current state of LayerRGBStream to a file.
     * @param filename  The name of the file to save.
     */
    void saveImage(const std::string &filename) const;

    /* The ImGui functions that will initialize the UI of the layer. */

    /**
     * Init UI for the camera settings.
     */
    void imGuiCameraSettings();

    /**
     * UI function to tell ImGui that the next ImGUI object will be rendered in the same line.
     */
    void imGuiSameLineSpace() const;

    /**
     * Render a rendering width button, that set the resolution width of the camera that renders the scene.
     * @param renderWidth       The width of the rendering to set.
     * @param buttonLabel       The label of the button.
     * @param afterPressWidth   The new width of the rendering.
     */
    void imGuiRenderWidthButton(int &renderWidth, const char *buttonLabel, int afterPressWidth) const;

    /**
     * Init the UI for the the Ray Tracer settings.
     */
    void imGuiRayTracerSettings();

    /**
     * Init the UI for the camera aperture slider.
     */
    void imGuiCameraApertureSlider();

    /**
     * Init the UI for the camera focus slider.
     */
    void imGuiCameraFocusDistSlider();

    /**
     * Init the UI for the camera Vertical field of view slider.
     */
    void imGuiCameraVFovSlider();

    /**
     * Init the UI for the number of render calls that will be performed in each loop of the main rendering loop.
     */
    void imGuiNRenderCallsSlider();

    /**
     * Init the UI buttons for size of the rendering width for the renderer.
     */
    void imGuiRenderWidthButtons();

    /**
     * Init the UI slider for the number of ray bounces.
     */
    void imGuiNRayBouncesSlider() const;

    /**
     * Init the UI section to save the image.
     */
    void imGuiSaveImageSection() const;

    /**
     * Init the UI that shows the current FPS info.
     */
    void imGuiFpsInfo();
};
