//
// Created by Sahar on 15/07/2022.
//

#pragma once


#include <vector>
#include <iostream>
#include "glew-2.1.0/include/GL/glew.h"

struct VertexBufferElement {
    unsigned int type;
    unsigned int count;
    unsigned char normalized;

    static unsigned int getSizeOfType(unsigned int type) {
        switch (type) {
            case GL_FLOAT:                return sizeof(float);
            case GL_UNSIGNED_BYTE:        return sizeof(unsigned char);
            case GL_UNSIGNED_INT:         return sizeof(unsigned int);
        }
        std::cerr << "Unsupported type: " << type << "\n";
        return 0;
    }
};

/**
 * Responsible for creating and initializing the buffer layout that will hold VertexBufferElements. When items are
 * pushed, will update the stride alignment of the vertex buffer according to the number of vertex elements and type.
 */
class VertexBufferLayout {
public:
    VertexBufferLayout() :stride(0) {}

    void push_float(unsigned int count) {
        elements.push_back({GL_FLOAT, count, GL_FALSE});
        stride += count * VertexBufferElement::getSizeOfType(GL_FLOAT);
    }

    void push_uint(unsigned int count) {
        elements.push_back({GL_UNSIGNED_INT, count, GL_FALSE});
        stride += count * VertexBufferElement::getSizeOfType(GL_UNSIGNED_INT);
    }

    void push_uchar(unsigned int count) {
        elements.push_back({GL_UNSIGNED_BYTE, count, GL_TRUE});
        stride += count * VertexBufferElement::getSizeOfType(GL_UNSIGNED_BYTE);
    }

    [[nodiscard]] unsigned int getStride() const {
        return stride;
    }

    [[nodiscard]] const std::vector<VertexBufferElement> &getElements() const {
        return elements;
    }

private:
    std::vector<VertexBufferElement> elements{};
    unsigned int stride;
};
