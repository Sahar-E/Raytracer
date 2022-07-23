//
// Created by Sahar on 14/07/2022.
//

#include "IndexBuffer.h"
#include "commonOpenGL.h"
#include "glew-2.1.0/include/GL/glew.h"


IndexBuffer::IndexBuffer(const unsigned int *data, unsigned int count)
    : _count(count) {
    static_assert(sizeof(unsigned int) == sizeof(GLuint));
    {
        checkGLErrors(glGenBuffers(1, &_rendererId));
        checkGLErrors(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _rendererId));
        checkGLErrors(glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(unsigned int), data, GL_STATIC_DRAW));
    }
}

IndexBuffer::~IndexBuffer() {
    checkGLErrors(glDeleteBuffers(1, &_rendererId));
}

void IndexBuffer::bind() const {
    checkGLErrors(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _rendererId));

}

void IndexBuffer::unbind() const {
    checkGLErrors(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
}
