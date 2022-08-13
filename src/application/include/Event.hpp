//
// Created by Sahar on 24/07/2022.
//

#pragma once

#include <string>
#include <functional>
#include <vector>
#include <map>
#include "common.h"


enum class EventType {
    None = 0,
    WindowClose, WindowResize, WindowFocus, WindowLostFocus, WindowMoved,
    KeyPressed, KeyReleased, KeyTyped,
    MouseButtonPressed, MouseButtonReleased, MouseMoved, MouseScrolled
};

enum EventCategory {
    None = 0,
    EventCategoryApplication = BIT(0),
    EventCategoryInput = BIT(1),
    EventCategoryKeyboard = BIT(2),
    EventCategoryMouse = BIT(3),
    EventCategoryMouseButton = BIT(4)
};

#define EVENT_CLASS_TYPE(type) static EventType getStaticType() { return EventType::type; }\
								virtual EventType getEventType() const override { return getStaticType(); }\
								virtual const char* getName() const override { return #type; }

#define EVENT_CLASS_CATEGORY(category) virtual int getCategoryFlags() const override { return category; }

/**
 * Class that represent an event that can be sent throughout the application.
 */
class Event {
public:
    virtual ~Event() = default;

    [[nodiscard]] virtual EventType getEventType() const = 0;

    [[nodiscard]] virtual const char* getName() const = 0;
    [[nodiscard]] virtual int getCategoryFlags() const = 0;
    [[nodiscard]] virtual std::string toString() const { return getName(); }
    [[nodiscard]] bool isInCategory(EventCategory category) const {
        return getCategoryFlags() & category;
    }

    [[nodiscard]] bool isHandled() const {
        return _isHandled;
    }

    void setHandled(bool handled) {
        _isHandled = handled;
    }

private:
    bool _isHandled = false;
};

/**
 * Templated dispatcher for events. Will be used to bind actions to events.
 */
class EventDispatcher
{
public:
    explicit EventDispatcher(Event& event) : m_Event(event) {}

    /**
     * Called when an event is handled.
     * @tparam T        EventType.
     * @tparam F        Function type to execute. Will be deduced by the compiler.
     * @param func      The function.
     * @return  Returns true if dispatched.
     */
    template<typename T, typename F>
    bool dispatch(const F& func)
    {
        if (m_Event.getEventType() == T::getStaticType())
        {
            m_Event.setHandled(m_Event.isHandled() | func(static_cast<T&>(m_Event)));
            return true;
        }
        return false;
    }
private:
    Event& m_Event;
};