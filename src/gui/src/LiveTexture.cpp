//
// Created by Sahar on 16/07/2022.
//

#include "LiveTexture.h"
#include "commonOpenGL.h"
#include "glew-2.1.0/include/GL/glew.h"



LiveTexture::LiveTexture(std::shared_ptr<unsigned char> buffer, int width, int height)
        : Texture("filepath-NA",
                  std::move(buffer),
                  width,
                  height,
                  -1) {
    checkGLErrors(glGenTextures(1, &_rendererId));
    checkGLErrors(glBindTexture(GL_TEXTURE_2D, _rendererId));

    checkGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    checkGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    checkGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    checkGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    checkGLErrors(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, _width, _height, 0,
                               GL_RGBA, GL_UNSIGNED_BYTE, _buffer.get()));
    checkGLErrors(glBindTexture(GL_TEXTURE_2D, 0));
}

void LiveTexture::updateTexture() {
//    checkGLErrors(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, _buffer.get()));
    checkGLErrors(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, _buffer.get()));

    checkGLErrors(glBindTexture(GL_TEXTURE_2D, 0));
}