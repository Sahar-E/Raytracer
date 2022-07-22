//
// Created by Sahar on 20/07/2022.
//

#pragma once

#include <string>
#include <memory>
#include "KeyInput.h"
#include "GLFW/glfw3.h"
#include "MouseInput.h"

class InputHandler {
public:

//    /**
//     * Must set the current window before calling this function.
//     * @return  The InputHandler instance.
//     */
//    static InputHandler &getInstance();

    explicit InputHandler(GLFWwindow* window);

    void handleKeyboardEvent(int key, bool isDown);
    [[nodiscard]] bool isKeyDown(int key) const;

    [[nodiscard]] bool isMouseMove() const;
    [[nodiscard]] double getMouseX() const;
    [[nodiscard]] double getMouseY() const;

private:

//    InputHandler(const InputHandler & application ) = delete;   // TODO-Sahar: ?
//    InputHandler &operator=(const InputHandler &instance) = delete;

    static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void keyDownCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static KeyInput * _keyInput;
    static MouseInput _mouseInput;
    GLFWwindow *_window;
};
