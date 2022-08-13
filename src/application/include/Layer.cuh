//
// Created by Sahar on 22/07/2022.
//

#pragma once

#include <string>
#include "Event.hpp"

/**
 * Layer for the application.
 */
class Layer {
public:
    Layer(const std::string& name = "Layer");
    virtual ~Layer() = default;

    /**
     * Runs when the layer gets attached to the Application.
     */
    virtual void onAttach() {}

    /**
     * Runs when the layer gets detached from the Application.
     */
    virtual void onDetach() {}

    /**
     * Updates on the main loop of the application.
     */
    virtual void onUpdate() {}

    /**
     * Runs when the application is sending an event.
     * @param event     The event sent.
     */
    virtual void onEvent(Event &event) {}

protected:
    std::string _layerName;
};
