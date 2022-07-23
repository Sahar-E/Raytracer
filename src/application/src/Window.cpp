//
// Created by Sahar on 20/07/2022.
//

#include "Window.h"
#include "InputHandler.h"
#include <iostream>
#include <utility>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"
#include "LiveTexture.h"
#include "VertexDrawer.h"
#include "commonOpenGL.h"
#include <imgui-docking/include/imgui.h>
#include <imgui-docking/include/imgui_impl_glfw.h>
#include <imgui-docking/include/imgui_impl_opengl3.h>


Window::Window(const std::string &name, float aspectRatio, int width) : _name(name), _aspectRatio(aspectRatio),
                                                                        _width(width), _glsl_version("#version 330") {
    initGLWindow();
    _inputHandler = std::make_shared<InputHandler>(_window);
    initGlBlendingConfigurations();
}

int Window::initGLWindow() {
    if (!glfwInit()) { return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);


    /* Create a windowed mode window and its OpenGL context */
    _window = glfwCreateWindow(_width, static_cast<int>(_width / _aspectRatio), _name.c_str(), nullptr, nullptr);
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

const std::shared_ptr<InputHandler> &Window::getInputHandler() const {
    return _inputHandler;
}
