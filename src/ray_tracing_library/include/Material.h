//
// Created by Sahar on 11/06/2022.
//

#pragma once


#include "HitResult.h"

struct HitResult;

class Material {
public:
    Material() = default;
    Material(const Color &albedo, double roughness) : _albedo(albedo), _roughness(roughness) {}

    [[nodiscard]]
    std::tuple<Color, Ray> getColorAndSecondaryRay(const HitResult &hitRes) const;

    [[nodiscard]]
    static Material getLambertian(const Color &albedo) {
        return {albedo, 1.0};
    }

private:
    Color _albedo;
    double _roughness{};

};