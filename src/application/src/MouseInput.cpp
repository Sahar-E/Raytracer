//
// Created by Sahar on 21/07/2022.
//

#include "MouseInput.h"

double MouseInput::getXPos() const {
    return _xPos;
}

void MouseInput::setXPos(double xPos) {
    MouseInput::_xPos = xPos;
}

double MouseInput::getYPos() const {
    return _yPos;
}

void MouseInput::setYPos(double yPos) {
    MouseInput::_yPos = yPos;
}
