//
// Created by Sahar on 11/06/2022.
//

#include "cuda_runtime_api.h"
#include "Material.cuh"
#include "Vec3.cuh"
#include "utils.cuh"

__device__ void Material::getColorAndSecondaryRay(const HitResult &hitRes, curandState *randState,
                                  Color &resEmittedColor, Color &resColor, Ray &resSecondaryRay) const {
    Vec3 secondaryRayDir{};
    Color resultColor{};

    Vec3 diffuseDir = normalize((hitRes.normal + randomUnitVec(randState)));
    double specularChance = randomFloatCuda(randState);
    if (specularChance < _percentSpecular) {
        getSpecularResult(hitRes, diffuseDir, specularChance, secondaryRayDir, resultColor, randState);
    } else {
        secondaryRayDir = diffuseDir;
        resultColor = _albedo;
    }

    resEmittedColor = _emittedColor;
    resColor = resultColor;
    resSecondaryRay = Ray(hitRes.hitPoint, secondaryRayDir);
}

__device__ void Material::getSpecularResult(const HitResult &hitRes, Vec3 &diffuseDir, double specularChance,
                                            Vec3 &resDir, Color &resColor, curandState *randState) const {
    double refractionIdxRatio = hitRes.isOutwardsNormal ? 1.0 / _refractionIdx : _refractionIdx;
    Vec3 rayDirNormalized = normalize(hitRes.hittingRay.direction());

    Vec3 specularDir{};
    bool doReflection = shouldDoReflection(hitRes, refractionIdxRatio, rayDirNormalized, randState);
    if (doReflection) {
        specularDir = reflect(hitRes.hittingRay.direction(), hitRes.normal);
    } else {
        specularDir = refract(rayDirNormalized, hitRes.normal, refractionIdxRatio);
    }

    resDir = normalize(alphaBlending(specularDir, diffuseDir, _roughnessSquared));
    resColor = alphaBlending(_specularColor, _albedo, specularChance);
}

__device__ bool Material::shouldDoReflection(const HitResult &hitRes, double refractionIdxRatio, Vec3 &rayDirNormalized,
                                             curandState *randState) const {
    double cosTheta = fmin(dot(-rayDirNormalized, hitRes.normal), 1.0);
    bool cannotRefract = cannotRefractBySnellsLaw(cosTheta, refractionIdxRatio);
    bool reflectApprox = reflectSchlickApproxForFrensel(cosTheta, refractionIdxRatio) > randomFloatCuda(randState);
    bool doReflection = !_isRefractable || cannotRefract || reflectApprox;
    return doReflection;
}
