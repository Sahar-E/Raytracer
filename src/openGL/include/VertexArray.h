//
// Created by Sahar on 15/07/2022.
//

#pragma once


#include "VertexBuffer.h"
#include "VertexBufferLayout.h"

/**
 * Class that is responsible to manage the binding of vertex buffers and their layout information.
 */
class VertexArray {
public:
    VertexArray();
    virtual ~VertexArray();
    VertexArray(const VertexArray &other) = delete;
    VertexArray &operator=(const VertexArray &other) = delete;

    void addBuffer(const VertexBuffer &vb, const VertexBufferLayout &layout);

    void bind() const;
    void unbind() const;

private:
    unsigned int rendererId;
};
