//
// Created by Sahar on 16/07/2022.
//

#pragma once

#include "Texture.h"
#include <memory>
#include "glew-2.1.0/include/GL/glew.h"

/**
 * LiveTexture will stream the buffer content as the texture to display.
 */
class LiveTexture : public Texture {
public:
    LiveTexture(std::shared_ptr<unsigned char[]> buffer, int width, int height, GLenum type);

    /**
     * Called to update the texture when it is changed.
     */
    void updateTexture();
};