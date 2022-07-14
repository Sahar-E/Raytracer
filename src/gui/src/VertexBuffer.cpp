//
// Created by Sahar on 14/07/2022.
//

#include "VertexBuffer.h"
#include "commonOpenGL.h"
#include "glew-2.1.0/include/GL/glew.h"


VertexBuffer::VertexBuffer(const void *data, unsigned int size) {
    checkGLErrors(glGenBuffers(1, &_rendererId));
    checkGLErrors(glBindBuffer(GL_ARRAY_BUFFER, _rendererId));
    checkGLErrors(glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW));

}

VertexBuffer::~VertexBuffer() {
    checkGLErrors(glDeleteBuffers(1, &_rendererId));
}

void VertexBuffer::bind() const {
    checkGLErrors(glBindBuffer(GL_ARRAY_BUFFER, _rendererId));

}

void VertexBuffer::unbind() const {
    checkGLErrors(glBindBuffer(GL_ARRAY_BUFFER, 0));
}
