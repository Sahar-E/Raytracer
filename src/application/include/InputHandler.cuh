//
// Created by Sahar on 20/07/2022.
//

#pragma once

#include <string>
#include <memory>
#include "glew-2.1.0/include/GL/glew.h"
#include "GLFW/glfw3.h"

class InputHandler {
public:
    [[nodiscard]] static bool isKeyDown(int key);

    [[nodiscard]] static double getMouseX();

    [[nodiscard]] static double getMouseY();

    [[nodiscard]] static bool isMousePressed(int button);

    [[nodiscard]] static std::pair<double, double> getMouseXY();

};
