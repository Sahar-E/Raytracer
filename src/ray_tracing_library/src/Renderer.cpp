//
// Created by Sahar on 10/06/2022.
//

#include <vector>
#include <Vec3.h>
#include <Renderer.h>
#include <omp.h>
#include <utils.h>

std::vector<Color> Renderer::render() const {
    std::vector<Color> data(_imageHeight * _imageWidth, {1, 1, 1});

#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 100) default(none) shared(std::cerr, data)
#endif
    for (int pixel_idx = 0; pixel_idx < _imageHeight * _imageWidth; ++pixel_idx) {
        int row = pixel_idx / _imageWidth;
        int col = pixel_idx % _imageWidth;
        if (omp_get_thread_num() == 0) {
            std::cerr << "\rRows left to render: " << _imageHeight - row << ' ' << std::flush;
        }
        Color pixelSum{};
        for (int i = 0; i < _nSamplesPerPixel; ++i) {
            auto h = (static_cast<double>(col) + randomDouble()) / (_imageWidth - 1);
            auto v = 1 - ((static_cast<double>(row) + randomDouble()) / (_imageHeight - 1));
            Ray ray = _camera.getRay(h, v);
            pixelSum += _world.rayTrace(ray, _nRayBounces);
        }
        Vec3 pixelColor = pixelSum / _nSamplesPerPixel;
        pixelColor = clamp(gammaCorrection(pixelColor), 0.0, 0.999);
        data[row * _imageWidth + col] = pixelColor;
    }
    return data;
}
