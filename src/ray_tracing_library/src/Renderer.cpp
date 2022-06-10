//
// Created by Sahar on 10/06/2022.
//

#include "Renderer.h"
#include <vector>


std::vector<Color> Renderer::start() const {
    Color black = {0, 0, 0};
    Color white = {1, 1, 1};
    std::vector<Color> data(_imageHeight * _imageWidth, black);

    for (int row = 0; row < _imageHeight; ++row) {
        for (int col = 0; col < _imageWidth; ++col) {
            if ((_imageWidth / 4 <= col && col <= _imageWidth / 2) && (_imageHeight / 4 <= row && row <= _imageHeight / 2)) {
                data[row * _imageWidth + col] = white;
            }
        }
    }
    return data;
}
