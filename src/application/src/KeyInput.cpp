//
// Created by Sahar on 20/07/2022.
//

#include "KeyInput.h"
#include <algorithm>


KeyInput::KeyInput(const std::vector<int>& keysToMonitor) {
    for (int key : keysToMonitor) {
        _keys[key] = false;
    }
}


bool KeyInput::getIsKeyDown(int key) {
    bool result = false;
    auto it = _keys.find(key);
    if (it != _keys.end()) {
        result = _keys[key];
    }
    return result;
}

void KeyInput::setIsKeyDown(int key, bool isDown) {
    auto it = _keys.find(key);
    if (it != _keys.end()) {
        _keys[key] = isDown;
    }
}


