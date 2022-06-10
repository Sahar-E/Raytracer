//
// Created by Sahar on 10/06/2022.
//

#include "Renderer.h"
#include <vector>
#include "utils.h"

std::vector<Color> Renderer::render() const {
    std::vector<Color> data(_imageHeight * _imageWidth, Color(BLACK_COLOR));

    for (int row = 0; row < _imageHeight; ++row) {
        for (int col = 0; col < _imageWidth; ++col) {
            auto h = static_cast<double>(col) / _imageWidth;
            auto v = static_cast<double>(row) / _imageHeight;
            Ray ray = _camera.getRay(h, v);
            data[row * _imageWidth + col] = _world.traceRay(ray);
        }
    }
    return data;
}
