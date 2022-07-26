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


class Window {
public:
    using EventCallbackFn = std::function<void(Event&)>;


    Window(const std::string &name, float aspectRatio, int width);

    [[nodiscard]] GLFWwindow *getWindow() const;

    [[nodiscard]] bool shouldClose() const;

    void setEventCallback(const EventCallbackFn& callback);

    [[nodiscard]] float getAspectRatio() const;

    [[nodiscard]] const std::string &getGlslVersion() const;

    void resizeWindow(int width, int height);

private:
    void onUpdate();
    int initGLWindow();
    static void initGlBlendingConfigurations() ;
    static void onWindowSizeChanged(GLFWwindow *window, int width, int height);

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
};
