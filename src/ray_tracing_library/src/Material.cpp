//
// Created by Sahar on 11/06/2022.
//

#include "Material.h"

bool Lambertian::getColor(const HitResult &hitRes, Color &attenuation, Ray &reflectionRay, Ray &refractionRay) const {
    attenuation = _albedo;
    Vec3 randomDirection = randomVecOnTangentSphere(hitRes.normal, hitRes.hitPoint) - hitRes.hitPoint;
    reflectionRay = Ray(hitRes.hitPoint, randomDirection);
    refractionRay = getZeroRay();
    return true;
}

Lambertian::Lambertian(const Color &albedo) : _albedo(albedo) {}


bool Metal::getColor(const HitResult &hitRes, Color &attenuation, Ray &reflectionRay, Ray &refractionRay) const {
    attenuation = _albedo;
    Vec3 randomDirection = randomVecOnTangentSphere(hitRes.normal, hitRes.hitPoint) - hitRes.hitPoint;
    Vec3 dirOfReflection = reflect(hitRes.hittingRay.direction(), hitRes.normal) + randomDirection * _fuzziness;

    reflectionRay = Ray(hitRes.hitPoint, dirOfReflection);
    refractionRay = getZeroRay();
    return true;
}
