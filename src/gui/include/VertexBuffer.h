//
// Created by Sahar on 14/07/2022.
//

#pragma once


class VertexBuffer {
public:
    VertexBuffer(const void *data, unsigned int size);
    virtual ~VertexBuffer();

    VertexBuffer(const VertexBuffer &other) = delete;
    VertexBuffer &operator=(const VertexBuffer &other) = delete;

    void bind() const;
    void unbind() const;

private:
    unsigned int _rendererId;

};
