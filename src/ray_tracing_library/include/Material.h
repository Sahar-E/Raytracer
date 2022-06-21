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
    std::tuple<Color, Color, Ray> getColorAndSecondaryRay(const HitResult &hitRes) const;

    [[nodiscard]]
    static Material getLambertian(const Color &albedo) {
        return Material(albedo);
    }

    [[nodiscard]]
    static Material
    getSpecular(const Color &albedo, const Color &specularColor, double roughness, double percentSpecular) {
        return {albedo, specularColor, roughness, percentSpecular};
    }

    [[nodiscard]]
    static Material getGlowing(const Color &albedo, const Color &emittedColor) {
        return {albedo, {0, 0, 0}, 0, 0, emittedColor};
    }

private:
    explicit Material(const Color &albedo) :
            Material(albedo, {0, 0, 0}, 0, 0) {}

    Material(const Color &albedo,
             const Color &specularColor,
             double roughnessSquared,
             double percentSpecular,
             const Color &emittedColor = {0, 0, 0})
            : _albedo(albedo),
              _specularColor(specularColor),
              _emittedColor(emittedColor),
              _percentSpecular(percentSpecular),
              _roughnessSquared(roughnessSquared * roughnessSquared) {}

    Color _albedo;
    Color _specularColor{};
    Color _emittedColor{};
    double _roughnessSquared{};
    double _percentSpecular{};
};