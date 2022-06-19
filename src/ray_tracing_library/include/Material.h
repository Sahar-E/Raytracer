//
// Created by Sahar on 11/06/2022.
//

#pragma once


#include "HitResult.h"

struct HitResult;

class Material {
public:
    [[nodiscard]] virtual bool getColor(const HitResult &hitRes,
                                        Color &attenuation,
                                        Ray &reflectionRay,
                                        Ray &refractionRay) const = 0;
};

class Lambertian : public Material {

public:

    explicit Lambertian(const Color &albedo) : _albedo(albedo) {}

    /**
     * // TODO-Sahar:
     * For Lambertian materials, the reflectionRay is the diffuse reflection from the surface.
     * There is no reflection from this material.
     *
     * @param hitRes
     * @param attenuation
     * @param reflectionRay
     * @param refractionRay
     * @return
     */
    bool getColor(const HitResult &hitRes, Color &attenuation, Ray &reflectionRay, Ray &refractionRay) const override;

private:
    Color _albedo;
};



class Metal : public Material {

public:
    Metal(const Color &albedo, double fuzziness) : _albedo(albedo), _roughness(fuzziness) {}

    bool getColor(const HitResult &hitRes, Color &attenuation, Ray &reflectionRay, Ray &refractionRay) const override;

private:
    Color _albedo;
    double _roughness;
};