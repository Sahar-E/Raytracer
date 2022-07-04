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

    __device__ void getColorAndSecondaryRay(const HitResult &hitRes, curandState *randState, Color &resEmittedColor,
                                            Color &resColor, Ray &resSecondaryRay) const;

    __host__ __device__ static Material getLambertian(const Color &albedo) {
        return Material(albedo);
    }

    __host__ __device__ static
    Material getSpecular(const Color &albedo, const Color &specularColor, float roughness, float percentSpecular) {
        return {albedo, specularColor, roughness, percentSpecular};
    }

    __host__ __device__ static
    Material getGlowing(const Color &albedo, const Color &emittedColor, float intensity) {
        return {albedo, emittedColor * intensity};
    }

    __host__ __device__ static
    Material getGlass(const Color &albedo, float refractionIdx) {
        return {albedo, refractionIdx};
    }

private:
    Color _albedo{};
    Color _specularColor{};
    Color _emittedColor{};
    float _percentSpecular{};
    float _roughnessSquared{};
    float _refractionIdx = 1.0f;
    bool _isRefractable{false};


    __host__ __device__
    explicit Material(const Color &albedo) : _albedo(albedo),
                                             _specularColor({0.0f, 0.0f, 0.0f}),
                                             _emittedColor({0.0f, 0.0f, 0.0f}),
                                             _percentSpecular(0.0f),
                                             _roughnessSquared(0.0f) {}

    __host__ __device__
    Material(const Color &albedo,
             const Color &specularColor,
             float roughness,
             float percentSpecular) : _albedo(albedo),
                                       _specularColor(specularColor),
                                       _emittedColor({0.0f, 0.0f, 0.0f}),
                                       _percentSpecular(percentSpecular),
                                       _roughnessSquared(roughness * roughness) {}

    __host__ __device__
    Material(const Color &albedo, const Color &emittedColor) : _albedo(albedo),
                                                               _specularColor({0.0f, 0.0f, 0.0f}),
                                                               _emittedColor(emittedColor),
                                                               _percentSpecular(0.0f),
                                                               _roughnessSquared(0.0f) {}

    __host__ __device__
    Material(const Color &albedo, float refractionIdx) : _albedo(albedo),
                                                          _specularColor({1.0f, 1.0f, 1.0f}),
                                                          _emittedColor({0.0f, 0.0f, 0.0f}),
                                                          _percentSpecular(1.0),
                                                          _roughnessSquared(0),
                                                          _refractionIdx(refractionIdx),
                                                          _isRefractable(true) {}


    __device__
    bool shouldDoReflection(const HitResult &hitRes, float refractionIdxRatio, Vec3 &rayDirNormalized,
                            curandState *randState) const;

    __device__
    void getSpecularResult(const HitResult &hitRes, Vec3 &diffuseDir, float specularChance, Vec3 &resDir,
                           Color &resColor, curandState *randState) const;
};