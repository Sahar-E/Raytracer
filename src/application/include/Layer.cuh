//
// Created by Sahar on 22/07/2022.
//

#pragma once

#include <string>
#include "Event.hpp"

class Layer {
public:
    Layer(const std::string& name = "Layer");
    virtual ~Layer() = default;

    virtual void onAttach() {}
    virtual void onDetach() {}
    virtual void onUpdate() {}
    virtual void onEvent(Event &event) {}

protected:
    std::string _layerName;
};
