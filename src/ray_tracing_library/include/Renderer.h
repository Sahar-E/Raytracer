//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include <vector>
#include <Vec3.hpp>

class Renderer {

public:

    Renderer(const int imageWidth, const int imageHeight) : _imageWidth(imageWidth), _imageHeight(imageHeight) {}

    [[nodiscard]] std::vector<Color> start() const;

private:
    int _imageWidth;
    int _imageHeight;
};



