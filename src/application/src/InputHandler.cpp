//
// Created by Sahar on 20/07/2022.
//

#include "InputHandler.h"

// TODO-Sahar: Bad design
KeyInput *InputHandler::_keyInput = new KeyInput({GLFW_KEY_A,
                                                         GLFW_KEY_D,
                                                         GLFW_KEY_W,
                                                         GLFW_KEY_S,
                                                         GLFW_KEY_SPACE,
                                                         GLFW_KEY_LEFT_SHIFT});
MouseInput InputHandler::_mouseInput;


void InputHandler::handleKeyboardEvent(int key, bool isDown) {
    _keyInput->setIsKeyDown(key, isDown);
}

void InputHandler::keyDownCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    _keyInput->setIsKeyDown(key, action != GLFW_RELEASE);
}

bool InputHandler::isKeyDown(int key)  const{
    return _keyInput->getIsKeyDown(key);
}

InputHandler::InputHandler(GLFWwindow* window) :_window(window){
    glfwSetKeyCallback(_window, InputHandler::keyDownCallback);
    glfwSetCursorPosCallback(_window, InputHandler::cursorPositionCallback);
    glfwSetMouseButtonCallback(_window, InputHandler::mouseButtonCallback);
}

void InputHandler::cursorPositionCallback(GLFWwindow *window, double xpos, double ypos) {
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        _mouseInput.setXPos(xpos);
        _mouseInput.setYPos(ypos);
    } else {
        _mouseInput.setXPos(0);
        _mouseInput.setYPos(0);
    }
}

bool InputHandler::isMouseMove() const {
    return _mouseInput.getXPos() != 0 || _mouseInput.getYPos() != 0;
}

double InputHandler::getMouseX() const {
    return _mouseInput.getXPos();
}

double InputHandler::getMouseY() const {
    return _mouseInput.getYPos();
}

void InputHandler::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
}
