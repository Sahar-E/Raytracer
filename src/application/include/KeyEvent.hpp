//
// Created by Sahar on 24/07/2022.
//

#pragma once

#include "Event.hpp"
#include <sstream>

class KeyEvent : public Event {
public:
    [[nodiscard]] int getCode() const { return _code; }

    EVENT_CLASS_CATEGORY(EventCategoryKeyboard | EventCategoryInput)

protected:
    explicit KeyEvent(const int keycode) : _code(keycode) {}

    int _code;
};

class KeyPressedEvent : public KeyEvent {
public:
    explicit KeyPressedEvent(const int keycode) : KeyEvent(keycode) {}

    [[nodiscard]] std::string toString() const override {
        std::stringstream ss;
        ss << "KeyPressedEvent: " << _code;
        return ss.str();
    }

    EVENT_CLASS_TYPE(KeyPressed)
};

class KeyReleasedEvent : public KeyEvent {
public:
    explicit KeyReleasedEvent(const int keycode) : KeyEvent(keycode) {}

    [[nodiscard]] std::string toString() const override {
        std::stringstream ss;
        ss << "KeyReleasedEvent: " << _code;
        return ss.str();
    }

    EVENT_CLASS_TYPE(KeyReleased)
};