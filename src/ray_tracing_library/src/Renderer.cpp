//
// Created by Sahar on 10/06/2022.
//

#include <vector>
#include <Vec3.h>
#include <Renderer.h>
#include <omp.h>

std::vector<Color> Renderer::render() const {
    std::vector<Color> data(_imageHeight * _imageWidth, {1, 1, 1});

#pragma omp parallel for schedule(dynamic) default(none) shared(std::cerr, data)
    for (int row = 0; row < _imageHeight; ++row) {
        if (omp_get_thread_num() == 0) {
            std::cerr << "\rRows left to render: " << _imageHeight - row << ' ' << std::flush;
        }
        for (int col = 0; col < _imageWidth; ++col) {
            Color pixelSum{};
            auto h = static_cast<double>(col) / (_imageWidth - 1);
            auto v = 1 - static_cast<double>(row) / (_imageHeight - 1);
//            auto h = (static_cast<double>(col) + randomDouble()) / (_imageWidth - 1);     // the random number is for simple Anti-aliasing
//            auto v = 1 - (static_cast<double>(row) + randomDouble()) / (_imageHeight - 1);
            Ray ray = _camera.getRay(h, v);
            for (int i = 0; i < _nSamplesPerPixel; ++i) {
                pixelSum += _world.rayTrace(ray, _nRayBounces);
            }
            data[row * _imageWidth + col] = pixelSum / _nSamplesPerPixel;
        }
    }
    return data;
}
