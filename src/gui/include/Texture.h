//
// Created by Sahar on 15/07/2022.
//

#pragma once

#include <string>

class Texture {
public:
    explicit Texture(const std::string &filepath);
    ~Texture();
    Texture(const Texture &other) = delete;
    Texture &operator=(const Texture &other) = delete;

    void bind(unsigned int slot = 0) const;
    void unbind() const;

    [[nodiscard]] int getWidth() const {
        return _width;
    }

    [[nodiscard]] int getHeight() const {
        return _height;
    }


private:
    unsigned int _rendererId;
    std::string _filepath;
    unsigned char *_localBuffer;
    int _width, _height, _bpp;
};
