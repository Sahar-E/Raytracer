//
// Created by Sahar on 15/07/2022.
//

#include "VertexArray.h"
#include "commonOpenGL.h"

VertexArray::VertexArray() {
    checkGLErrors(glGenVertexArrays(1, &rendererId));
}

VertexArray::~VertexArray() {
    checkGLErrors(glDeleteVertexArrays(1, &rendererId));
}


void VertexArray::addBuffer(const VertexBuffer &vb, const VertexBufferLayout &layout) {
    bind();
    vb.bind();
    const auto& elements = layout.getElements();
    char * offset = 0;
    for( unsigned int i = 0; i < elements.size(); ++i ) {
        const auto& element = elements[i];
        checkGLErrors(glEnableVertexAttribArray(i));
        checkGLErrors(glVertexAttribPointer(i, element.count, element.type, element.normalized, layout.getStride(), offset));
        offset += element.count * VertexBufferElement::getSizeOfType(element.type);
    }

}

void VertexArray::bind() const {
    checkGLErrors(glBindVertexArray(rendererId));
}

void VertexArray::unbind() const {
    checkGLErrors(glBindVertexArray(0));
}
