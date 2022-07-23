//
// Created by Sahar on 16/07/2022.
//

#include "LiveTexture.h"

#include <utility>
#include "commonOpenGL.h"
#include "glew-2.1.0/include/GL/glew.h"
#include "../../ray_tracing_library/include/TimeThis.h"


LiveTexture::LiveTexture(std::shared_ptr<unsigned char[]> buffer, int width, int height, GLenum type)
        : Texture(std::move(buffer), width, height, type){
    checkGLErrors(glGenTextures(1, &_rendererId));
    checkGLErrors(glBindTexture(GL_TEXTURE_2D, _rendererId));

    checkGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    checkGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    checkGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    checkGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    checkGLErrors(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, _width, _height, 0,
                               _type, GL_UNSIGNED_BYTE, _buffer.get()));
    checkGLErrors(glBindTexture(GL_TEXTURE_2D, 0));
}

void LiveTexture::updateTexture() {
    checkGLErrors(glBindTexture(GL_TEXTURE_2D, _rendererId));
    checkGLErrors(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, _type, GL_UNSIGNED_BYTE, _buffer.get()));

}
