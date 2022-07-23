//
// Created by Sahar on 20/07/2022.
//

#pragma once


#include "glew-2.1.0/include/GL/glew.h"
#include "GLFW/glfw3.h"
#include "InputHandler.h"
#include <memory>
#include <string>


class Window {
public:
    Window(const std::string &name, float aspectRatio, int width);

    [[nodiscard]] GLFWwindow *getWindow() const;

    bool shouldClose() const;

    const std::shared_ptr<InputHandler> &getInputHandler() const;

    float getAspectRatio() const;

    const std::string &getGlslVersion() const;

private:
    void onUpdate();
    int initGLWindow();
    static void initGlBlendingConfigurations() ;

private:
    GLFWwindow *_window;
    const std::string &_name;
    float _aspectRatio;
    int _width;
    std::string _glsl_version;
    std::shared_ptr<InputHandler> _inputHandler;
};
