//
// Created by Sahar on 20/07/2022.
//

#pragma once


#include "glew-2.1.0/include/GL/glew.h"
#include "GLFW/glfw3.h"
#include "InputHandler.cuh"
#include "Event.hpp"
#include <memory>
#include <string>
#include <algorithm>


/**
 * Represent the main window of the application.
 */
class Window {
public:
    using EventCallbackFn = std::function<void(Event&)>;

    /**
     * Constructor for the main window of the application.
     * @param name          The name of the window, will be displayed in the title bar.
     * @param aspectRatio   The aspectRatio of the window.
     * @param width         The width of the window.
     */
    Window(const std::string &name, float aspectRatio, int width);

    [[nodiscard]] GLFWwindow *getWindow() const;

    /**
     * @return  True if should close the window. For example, will be true if the user press the exit button.
     */
    [[nodiscard]] bool shouldClose() const;

    /**
     * Sets a callback function to be called when the there is an event that is sent in the window.
     * @param callback  The callback function.
     */
    void setEventCallback(const EventCallbackFn& callback);

    [[nodiscard]] float getAspectRatio() const;

    [[nodiscard]] const std::string &getGlslVersion() const;

    /**
     * Setter for width and height of the window.
     * @param width     The new width of the window.
     * @param height    The new height of the window.
     */
    void resizeWindow(int width, int height);

private:
    GLFWwindow *_window;
    std::string _glsl_version;

    struct WindowData
    {
        std::string title;
        unsigned int width, height;
        EventCallbackFn eventCallback;
    };
    WindowData _data;

    void setGlfwCallbacksEvents();

    void setGlfwSetWindowSizeCallback() const;

    void setGlfwSetKeyCallback() const;

    void setGlfwSetMouseButtonCallback() const;

    void setGlfwSetCursorPosCallback() const;

    void onUpdate();

    int initGLWindow();

    static void initGlBlendingConfigurations();

};
