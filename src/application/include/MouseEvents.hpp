//
// Created by Sahar on 24/07/2022.
//

#pragma once

#include "Event.hpp"
#include <sstream>

class MouseMovedEvent : public Event {
public:
    MouseMovedEvent(const float x, const float y)
            : _mouseX(x), _mouseY(y) {}

    float getX() const { return _mouseX; }

    float getY() const { return _mouseY; }

    std::string toString() const override {
        std::stringstream ss;
        ss << "MouseMovedEvent: " << _mouseX << ", " << _mouseY;
        return ss.str();
    }

    EVENT_CLASS_TYPE(MouseMoved)

    EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput)

private:
    float _mouseX, _mouseY;
};


class MouseButtonEvent : public Event {
public:
    int getMouseButton() const { return _button; }

    EVENT_CLASS_CATEGORY(EventCategoryMouse | EventCategoryInput | EventCategoryMouseButton)

protected:
    explicit MouseButtonEvent(const int button)
            : _button(button) {}

    int _button;
};

class MouseButtonPressedEvent : public MouseButtonEvent {
public:
    explicit MouseButtonPressedEvent(const int button)
            : MouseButtonEvent(button) {}

    [[nodiscard]] std::string toString() const override {
        std::stringstream ss;
        ss << "MouseButtonPressedEvent: " << _button;
        return ss.str();
    }

    EVENT_CLASS_TYPE(MouseButtonPressed)
};

class MouseButtonReleasedEvent : public MouseButtonEvent {
public:
    explicit MouseButtonReleasedEvent(const int button)
            : MouseButtonEvent(button) {}

    [[nodiscard]] std::string toString() const override {
        std::stringstream ss;
        ss << "MouseButtonReleasedEvent: " << _button;
        return ss.str();
    }

    EVENT_CLASS_TYPE(MouseButtonReleased)
};