//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include <vector>
#include <Vec3.hpp>
#include "World.h"
#include "Camera.h"

class Renderer {

public:

    Renderer(const int imageWidth, const int imageHeight, const World &world, const Camera &camera)
            : _imageWidth(imageWidth), _imageHeight(imageHeight), _world(world), _camera(camera) {}

    [[nodiscard]] std::vector<Color> render() const;

private:
    int _imageWidth;
    int _imageHeight;
    const World &_world;
    const Camera &_camera;
};



