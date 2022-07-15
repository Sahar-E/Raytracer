//
// Created by Sahar on 15/07/2022.
//

#include "Texture.h"
#include "commonOpenGL.h"
#include "glew-2.1.0/include/GL/glew.h"
#include "stb_library/include/stb_library/stb_image.h"

Texture::Texture(const std::string &filepath)
    : _rendererId(0), _filepath(filepath), _localBuffer(nullptr), _width(0), _height(0), _bpp(0) {

    stbi_set_flip_vertically_on_load(true);
    _localBuffer = stbi_load(filepath.c_str(), &_width, &_height, &_bpp, 4);

    checkGLErrors(glGenTextures(1, &_rendererId));
    checkGLErrors(glBindTexture(GL_TEXTURE_2D, _rendererId));

    checkGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    checkGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    checkGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    checkGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    checkGLErrors(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, _width, _height, 0,
                                 GL_RGBA, GL_UNSIGNED_BYTE, _localBuffer));
    checkGLErrors(glBindTexture(GL_TEXTURE_2D, 0));

    if (_localBuffer) {
        stbi_image_free(_localBuffer);
    }
}

Texture::~Texture() {
    checkGLErrors(glDeleteTextures(1, &_rendererId));
}

void Texture::bind(unsigned int slot) const {
    checkGLErrors(glActiveTexture(GL_TEXTURE0 + slot));
    checkGLErrors(glBindTexture(GL_TEXTURE_2D, _rendererId));
}

void Texture::unbind() const {
    checkGLErrors(glBindTexture(GL_TEXTURE_2D, 0));
}
