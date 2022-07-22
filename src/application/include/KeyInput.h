//
// Created by Sahar on 20/07/2022.
//

#pragma once

#include <vector>
#include <map>

class KeyInput {
public:
    KeyInput(const std::vector<int>& keysToMonitor);
    // If this KeyInput is enabled and the given key is monitored,
    // returns pressed state.  Else returns false.
    bool getIsKeyDown(int key);

    // Used internally to update key states.  Called by the GLFW callback.
    void setIsKeyDown(int key, bool isDown);
private:
    // Map from monitored keyes to their pressed states
    std::map<int, bool> _keys;
};
