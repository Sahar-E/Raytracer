//
// Created by Sahar on 24/07/2022.
//

#pragma once

#include <sstream>
#include "Event.hpp"

class WindowResizeEvent : public Event
{
public:
    WindowResizeEvent(unsigned int width, unsigned int height)
            : m_Width(width), m_Height(height) {}

    [[nodiscard]] unsigned int getWidth() const { return m_Width; }
    [[nodiscard]] unsigned int getHeight() const { return m_Height; }

    [[nodiscard]] std::string toString() const override
    {
        std::stringstream ss;
        ss << "WindowResizeEvent: " << m_Width << ", " << m_Height;
        return ss.str();
    }

    EVENT_CLASS_TYPE(WindowResize)
    EVENT_CLASS_CATEGORY(EventCategoryApplication)
private:
    unsigned int m_Width, m_Height;
};
