//
// Created by Sahar on 11/06/2022.
//

#include <utils.h>
#include "Material.h"


std::tuple<Color, Color, Ray> Material::getColorAndSecondaryRay(const HitResult &hitRes) const {
    Vec3 secondaryRayDir;
    Vec3 diffuseDir = normalize((hitRes.normal + randomUnitVec()));
    Color resultColor;

    double specularChance = randomDouble();
    bool shouldDoSpecular = specularChance < _percentSpecular;
    if (shouldDoSpecular) {
        Vec3 specularDir = reflect(hitRes.hittingRay.direction(), hitRes.normal);
        specularDir = normalize(alphaBlending(specularDir, diffuseDir, _roughnessSquared));
        secondaryRayDir = specularDir;
        resultColor = alphaBlending(_specularColor, _albedo, specularChance);
    } else {
        secondaryRayDir = diffuseDir;
        resultColor = _albedo;
    }

    Ray secondaryRay = Ray(hitRes.hitPoint, secondaryRayDir);
    return {_emittedColor, resultColor, secondaryRay};
}

//bool Glass::getColor(const HitResult &hitRes, Color &attenuation, Ray &reflectionRay, Ray &refractionRay) const {
//    attenuation = _albedo;
//    double hitRay_dot_normal = dot(hitRes.hittingRay.direction(), hitRes.normal);
//    bool front_face = hitRay_dot_normal < 0;
//    double refraction_ratio = front_face ? (1.0 / _refractiveIndex) : _refractiveIndex;
//
//    Vec3 unit_direction = unitVector(hitRes.hittingRay.direction());
//    double cos_theta = fmin(-hitRay_dot_normal, 1.0);
//    double sin_theta = sqrt(1.0 - cos_theta*cos_theta);
//
//    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
//    Vec3 direction;
//    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > randomDouble()) {
//        direction = reflect(unit_direction, hitRes.normal);
//    } else {
//        direction = refract(unit_direction, hitRes.normal, refraction_ratio);
//    }
//    refractionRay = Ray(hitRes.hitPoint, direction);
//    return true;
//}
