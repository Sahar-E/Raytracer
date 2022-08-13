//
// Created by Sahar on 20/07/2022.
//

#pragma once

#include <string>
#include <memory>
#include "glew-2.1.0/include/GL/glew.h"
#include "GLFW/glfw3.h"

/**
 * This class can be used to get information about mouse and keyboard states.
 */
class InputHandler {
public:
    [[nodiscard]] static bool isKeyDown(int keyType);

    [[nodiscard]] static double getMouseX();

    [[nodiscard]] static double getMouseY();

    [[nodiscard]] static bool isMousePressed(int buttonType);

    [[nodiscard]] static std::pair<double, double> getMouseXY();
};
