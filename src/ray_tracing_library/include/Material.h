//
// Created by Sahar on 11/06/2022.
//

#pragma once


#include "HitResult.h"

struct HitResult;

class Material {
public:
    Material() = default;

    [[nodiscard]]
    std::tuple<Color, Ray> getColorAndSecondaryRay(const HitResult &hitRes) const;

    [[nodiscard]]
    static Material getLambertian(const Color &albedo) {
        return Material(albedo);
    }

    [[nodiscard]]
    static Material getSpecular(const Color &albedo, double roughness, double percentSpecular) {
        return {albedo, roughness, percentSpecular};
    }

private:
    explicit Material(const Color &albedo) : Material(albedo, 0, 0) {}

    Material(const Color &albedo,
             double roughnessSquared,
             double percentSpecular) :
            _albedo(albedo), _roughnessSquared(roughnessSquared * roughnessSquared),
            _percentSpecular(percentSpecular), _emissiveLight(0) {}

    Color _albedo;
    double _roughnessSquared{};
    double _percentSpecular{};
    double _emissiveLight{};      // TODO-Sahar: add const.
};