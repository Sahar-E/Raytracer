//
// Created by Sahar on 24/07/2022.
//

#pragma once

#include <sstream>
#include "Event.hpp"

/**
 * Sent when the application window is resized.
 */
class WindowResizeEvent : public Event {
public:
    WindowResizeEvent(int width, int height)
            : m_Width(width), m_Height(height) {}

    [[nodiscard]] int getWidth() const { return m_Width; }

    [[nodiscard]] int getHeight() const { return m_Height; }

    [[nodiscard]] std::string toString() const override {
        std::stringstream ss;
        ss << "WindowResizeEvent: " << m_Width << ", " << m_Height;
        return ss.str();
    }

    EVENT_CLASS_TYPE(WindowResize)

    EVENT_CLASS_CATEGORY(EventCategoryApplication)

private:
    int m_Width, m_Height;
};
