//
// Created by Sahar on 11/06/2022.
//

#pragma once


#include "HitResult.h"

struct HitResult;

/**
 * Represent a material property for the renderer.
 *
 *
 * It is not implemented with OOP design in order to gain performance.
 */
class Material {
public:
    Material() = default;

    [[nodiscard]] std::tuple<Color, Color, Ray> getColorAndSecondaryRay(const HitResult &hitRes) const;

    [[nodiscard]] static Material getLambertian(const Color &albedo) {
        return Material(albedo);
    }

    [[nodiscard]] static Material
    getSpecular(const Color &albedo, const Color &specularColor, double roughness, double percentSpecular) {
        return {albedo, specularColor, roughness, percentSpecular};
    }

    [[nodiscard]] static Material getGlowing(const Color &albedo, const Color &emittedColor) {
        return {albedo, emittedColor};
    }

    [[nodiscard]] static Material getGlass(const Color &albedo, double refractionIdx) {
        return {albedo, refractionIdx};
    }

private:
    Color _albedo{};
    Color _specularColor{};
    Color _emittedColor{};
    double _percentSpecular{};
    double _roughnessSquared{};
    double _refractionIdx = 1.0;
    bool _isRefractable = false;


    explicit Material(const Color &albedo) : _albedo(albedo),
                                             _specularColor({0, 0, 0}),
                                             _emittedColor({0, 0, 0}),
                                             _percentSpecular(0),
                                             _roughnessSquared(0) {}

    Material(const Color &albedo,
             const Color &specularColor,
             double roughness,
             double percentSpecular) : _albedo(albedo),
                                       _specularColor(specularColor),
                                       _emittedColor({0, 0, 0}),
                                       _percentSpecular(percentSpecular),
                                       _roughnessSquared(roughness * roughness) {}

    Material(const Color &albedo, const Color &emittedColor) : _albedo(albedo),
                                                               _specularColor({0, 0, 0}),
                                                               _emittedColor(emittedColor),
                                                               _percentSpecular(0),
                                                               _roughnessSquared(0) {}

    Material(const Color &albedo, double refractionIdx) : _albedo(albedo),
                                                          _specularColor({1, 1, 1}),
                                                          _emittedColor({0, 0, 0}),
                                                          _percentSpecular(1.0),
                                                          _roughnessSquared(0),
                                                          _refractionIdx(refractionIdx),
                                                          _isRefractable(true) {}


    bool shouldDoReflection(const HitResult &hitRes, double refractionIdxRatio, Vec3 &rayDirNormalized) const;

    std::tuple<Vec3, Color> getSpecularResult(const HitResult &hitRes, Vec3 &diffuseDir, double specularChance) const;
};