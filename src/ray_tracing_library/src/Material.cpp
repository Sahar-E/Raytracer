//
// Created by Sahar on 11/06/2022.
//

#include "utils.h"
#include "Material.h"


std::tuple<Color, Color, Ray> Material::getColorAndSecondaryRay(const HitResult &hitRes) const {
    Vec3 secondaryRayDir;
    Color resultColor;
    Vec3 diffuseDir = normalize((hitRes.normal + randomUnitVec()));

    double specularChance = randomDouble();
    if (specularChance < _percentSpecular) {
        std::tie(secondaryRayDir,resultColor ) = getSpecularResult(hitRes, diffuseDir, specularChance);
    } else {
        secondaryRayDir = diffuseDir;
        resultColor = _albedo;
    }

    Ray secondaryRay = Ray(hitRes.hitPoint, secondaryRayDir);
    return {_emittedColor, resultColor, secondaryRay};
}

std::tuple<Vec3, Color> Material::getSpecularResult(const HitResult &hitRes,
                                                    Vec3 &diffuseDir,
                                                    double specularChance) const {
    double refractionIdxRatio = hitRes.isOutwardsNormal ? 1.0 / _refractionIdx : _refractionIdx;
    Vec3 rayDirNormalized = normalize(hitRes.hittingRay.direction());

    Vec3 specularDir;
    bool doReflection = shouldDoReflection(hitRes, refractionIdxRatio, rayDirNormalized);
    if (doReflection) {
        specularDir = reflect(hitRes.hittingRay.direction(), hitRes.normal);
    } else {
        specularDir = refract(rayDirNormalized, hitRes.normal, refractionIdxRatio);
    }

    specularDir = normalize(alphaBlending(specularDir, diffuseDir, _roughnessSquared));
    Color color = alphaBlending(_specularColor, _albedo, specularChance);
    return {specularDir, color};
}

bool Material::shouldDoReflection(const HitResult &hitRes, double refractionIdxRatio, Vec3 &rayDirNormalized) const {
    double cosTheta = fmin(dot(-rayDirNormalized, hitRes.normal), 1.0);
    bool cannotRefract = cannotRefractBySnellsLaw(cosTheta, refractionIdxRatio);
    bool reflectBySchlickApprox = reflectSchlickApproxForFrensel(cosTheta, refractionIdxRatio) > randomDouble();
    bool doReflection = !_isRefractable || cannotRefract || reflectBySchlickApprox;
    return doReflection;
}
