//
// Created by Sahar on 11/06/2022.
//

#include "Material.h"


std::tuple<Color, Ray> Material::getColorAndSecondaryRay(const HitResult &hitRes) const {
    Vec3 randomDirection = randomVecOnTangentSphere(hitRes.normal, hitRes.hitPoint) - hitRes.hitPoint;
    Vec3 dirOfReflection;
    if (_roughness > 0.99f) {
        dirOfReflection = randomDirection;
    } else {
        dirOfReflection = unitVector(reflect(hitRes.hittingRay.direction(), hitRes.normal)) +
                          unitVector(randomDirection) * _roughness;
    }
    Ray secondaryRay = Ray(hitRes.hitPoint, dirOfReflection);
    return {_albedo, secondaryRay};
}
