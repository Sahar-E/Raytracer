//
// Created by Sahar on 11/06/2022.
//

#pragma once


#include "HitResult.h"

/**
 * Represent a material property for the renderer.
 *
 *
 * It is not implemented with OOP design in order to gain performance.
 */
class Material {
public:
    __host__ __device__
    Material() = default;

    __host__ __device__ void
    getColorAndSecondaryRay(const HitResult &hitRes, int &randState, Color &resEmittedColor, Color &resColor,
                            Ray &resSecondaryRay) const;

    __host__ __device__ static Material getLambertian(const Color &albedo) {
        return Material(albedo);
    }

    __host__ __device__ static
    Material getSpecular(const Color &albedo, const Color &specularColor, double roughness, double percentSpecular) {
        return {albedo, specularColor, roughness, percentSpecular};
    }

    __host__ __device__ static
    Material getGlowing(const Color &albedo, const Color &emittedColor, double intensity) {
        return {albedo, emittedColor * intensity};
    }

    __host__ __device__ static
    Material getGlass(const Color &albedo, double refractionIdx) {
        return {albedo, refractionIdx};
    }

private:
    Color _albedo{};
    Color _specularColor{};
    Color _emittedColor{};
    double _percentSpecular{};
    double _roughnessSquared{};
    double _refractionIdx = 1.0;
    bool _isRefractable{false};


    __host__ __device__
    explicit Material(const Color &albedo) : _albedo(albedo),
                                             _specularColor({0, 0, 0}),
                                             _emittedColor({0, 0, 0}),
                                             _percentSpecular(0),
                                             _roughnessSquared(0) {}

    __host__ __device__
    Material(const Color &albedo,
             const Color &specularColor,
             double roughness,
             double percentSpecular) : _albedo(albedo),
                                       _specularColor(specularColor),
                                       _emittedColor({0, 0, 0}),
                                       _percentSpecular(percentSpecular),
                                       _roughnessSquared(roughness * roughness) {}

    __host__ __device__
    Material(const Color &albedo, const Color &emittedColor) : _albedo(albedo),
                                                               _specularColor({0, 0, 0}),
                                                               _emittedColor(emittedColor),
                                                               _percentSpecular(0),
                                                               _roughnessSquared(0) {}

    __host__ __device__
    Material(const Color &albedo, double refractionIdx) : _albedo(albedo),
                                                          _specularColor({1, 1, 1}),
                                                          _emittedColor({0, 0, 0}),
                                                          _percentSpecular(1.0),
                                                          _roughnessSquared(0),
                                                          _refractionIdx(refractionIdx),
                                                          _isRefractable(true) {}


    __host__ __device__
    bool shouldDoReflection(const HitResult &hitRes, double refractionIdxRatio, Vec3 &rayDirNormalized,
                            int &randState) const;

    __host__ __device__
    void getSpecularResult(const HitResult &hitRes, Vec3 &diffuseDir, double specularChance, Vec3 &resDir,
                           Color &resColor, int &randState) const;
};