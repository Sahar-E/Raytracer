//
// Created by Sahar on 20/07/2022.
//

#include "Window.h"
#include "InputHandler.cuh"
#include <iostream>
#include "Shader.h"
#include "LiveTexture.h"
#include "commonOpenGL.h"
#include "ApplicationEvents.hpp"
#include "KeyEvent.hpp"
#include "MouseEvents.hpp"


void Window::onWindowSizeChanged(GLFWwindow *window, int width, int height) {

}

Window::Window(const std::string &name, float aspectRatio, int width) : _glsl_version("#version 330") {
    _data.title = name;
    _data.width = width;
    _data.height = width / aspectRatio;

    initGLWindow();
    initGlBlendingConfigurations();
    setGlfwCallbacksEvents();
}

void Window::setGlfwCallbacksEvents() {
    glfwSetWindowUserPointer(_window, &_data);
    setGlfwSetWindowSizeCallback();
    setGlfwSetKeyCallback();
    setGlfwSetMouseButtonCallback();
    setGlfwSetCursorPosCallback();
}

void Window::setGlfwSetCursorPosCallback() const {
    glfwSetCursorPosCallback(_window, [](GLFWwindow *window, double xpos, double ypos) {
        WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

        MouseMovedEvent event(static_cast<float>(xpos), static_cast<float>(ypos));
        data.eventCallback(event);
    });
}

void Window::setGlfwSetMouseButtonCallback() const {
    glfwSetMouseButtonCallback(_window, [](GLFWwindow *window, int button, int action, int mods) {
        WindowData &data = *(WindowData *) glfwGetWindowUserPointer(window);
        switch (action)
        {
            case GLFW_PRESS:
            {
                MouseButtonPressedEvent event(button);
                data.eventCallback(event);
                break;
            }
            case GLFW_RELEASE:
            {
                MouseButtonReleasedEvent event(button);
                data.eventCallback(event);
                break;
            }
            default:
                break;
        }
    });
}

void Window::setGlfwSetKeyCallback() const {
    glfwSetKeyCallback(_window, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
        WindowData &data = *(WindowData *) glfwGetWindowUserPointer(window);

        switch (action)
        {
            case GLFW_PRESS:
            case GLFW_REPEAT:
            {
                KeyPressedEvent event(key);
                data.eventCallback(event);
                break;
            }
            case GLFW_RELEASE:
            {
                KeyReleasedEvent event(key);
                data.eventCallback(event);
                break;
            }
            default:
                break;
        }
    });
}

void Window::setGlfwSetWindowSizeCallback() const {
    glfwSetWindowSizeCallback(_window, [](GLFWwindow *window, int width, int height) {
        WindowData &data = *(WindowData *) glfwGetWindowUserPointer(window);
        data.width = width;
        data.height = height;

        WindowResizeEvent event(width, height);
        data.eventCallback(event);
    });
}

int Window::initGLWindow() {
    if (!glfwInit()) { return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);


    /* Create a windowed mode window and its OpenGL context */
    _window = glfwCreateWindow(_data.width, _data.height, _data.title.c_str(), nullptr, nullptr);
    if (!_window) {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(_window);

    glfwSwapInterval(1);

    if (glewInit() != GLEW_OK) {
        std::cerr << "glewInit() failed\n";
    }


    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    return 0;
}

void Window::onUpdate() {}

void Window::initGlBlendingConfigurations() {
    checkGLErrors(glEnable(GL_BLEND));
    checkGLErrors(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
}

GLFWwindow *Window::getWindow() const {
    return _window;
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(_window);
}

const std::string &Window::getGlslVersion() const {
    return _glsl_version;
}

float Window::getAspectRatio() const {
    return static_cast<float>(_data.width) / _data.height;
}

void Window::setEventCallback(const Window::EventCallbackFn &callback) {
    _data.eventCallback = callback;
}

