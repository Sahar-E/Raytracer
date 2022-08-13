//
// Created by Sahar on 20/07/2022.
//

#include "InputHandler.cuh"
#include "Application.cuh"


bool InputHandler::isKeyDown(int keyType) {
    auto state = glfwGetKey(Application::getInstance().getWindow()->getWindow(), keyType);
    return state == GLFW_PRESS || state == GLFW_REPEAT;
}

bool InputHandler::isMousePressed(int buttonType) {
    auto state = glfwGetMouseButton(Application::getInstance().getWindow()->getWindow(), buttonType);
    return state == GLFW_PRESS;
}


double InputHandler::getMouseX() {
    double xpos, ypos;
    glfwGetCursorPos(Application::getInstance().getWindow()->getWindow(), &xpos, &ypos);
    return xpos;
}

double InputHandler::getMouseY() {
    double xpos, ypos;
    glfwGetCursorPos(Application::getInstance().getWindow()->getWindow(), &xpos, &ypos);
    return ypos;
}

std::pair<double, double> InputHandler::getMouseXY() {
    double xpos, ypos;
    glfwGetCursorPos(Application::getInstance().getWindow()->getWindow(), &xpos, &ypos);
    return std::make_pair(xpos, ypos);
}

