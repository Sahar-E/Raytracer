//
// Created by Sahar on 14/07/2022.
//

#pragma once


class IndexBuffer {
public:
    IndexBuffer(const unsigned int *data, unsigned int count);
    virtual ~IndexBuffer();

    void bind() const;
    void unbind() const;


    [[nodiscard]] unsigned int getCount() const {
        return _count;
    }

private:
    unsigned int _rendererId;
    unsigned int _count;

};
