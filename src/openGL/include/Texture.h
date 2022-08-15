//
// Created by Sahar on 15/07/2022.
//

#pragma once

#include <string>
#include <memory>
#include "glew-2.1.0/include/GL/glew.h"

/**
 * Class that handles the texture for OpenGL rendering.
 */
class Texture {
public:
    explicit Texture(const std::string &filepath);
    virtual ~Texture();

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

protected:
    Texture(std::shared_ptr<unsigned char[]> buffer, int width, int height, GLenum type);

protected:
    unsigned int _rendererId;
    std::string _filepath;  // For debug purposes.
    std::shared_ptr<unsigned char[]> _buffer;
    int _width, _height, _channelsInFile;
    GLenum _type;
};

