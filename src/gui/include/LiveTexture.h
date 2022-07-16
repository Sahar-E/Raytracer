//
// Created by Sahar on 16/07/2022.
//

#pragma once

#include "Texture.h"
#include <memory>

class LiveTexture : public Texture {
public:
    LiveTexture(std::shared_ptr<unsigned char> buffer, int width, int height);
    void updateTexture();

};