//
// Created by Sahar on 14/07/2022.
//

#pragma once


/**
 * Class that responsible for handling the indices that will be used in OpenGL rendering of the VertexBuffer.
 */
class IndexBuffer {
public:
    IndexBuffer(const unsigned int *data, unsigned int count);
    virtual ~IndexBuffer();

    IndexBuffer(const IndexBuffer &other) = delete;
    IndexBuffer &operator=(const IndexBuffer &other) = delete;

    void bind() const;
    void unbind() const;


    [[nodiscard]] unsigned int getCount() const {
        return _count;
    }

private:
    unsigned int _rendererId;
    unsigned int _count;
};
