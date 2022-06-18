//
// Created by Sahar on 10/06/2022.
//

#include <HitResult.h>
#include <constants.h>
#include <utils.h>
#include "World.h"

Color World::rayTrace(const Ray &ray, int bounce) const {
    const static Color black = {0, 0, 0};
    if (bounce <= 0) {
        return black;
    }
    HitResult hitRes;
    bool hit = getHitResult(ray, hitRes);

    if (hit) {
        Color attenuation;
        Ray reflectionRay;
        Ray refractionRay;
//        isZeroVec() // TODO-Sahar: use this.
        if (hitRes.material->getColor(hitRes, attenuation, reflectionRay, refractionRay)) {
            Color scatterColor = rayTrace(reflectionRay, bounce - 1) * attenuation;
            return scatterColor;
        } else {
            return black;
        }
//        const Ray &reflectionRay = hitRes.reflectionRay;
//        return rayTrace(reflectionRay, bounce - 1) * 0.5;
    } else {
        return World::backgroundColor(ray);
    }
}

bool World::getHitResult(const Ray &ray, HitResult &hitRes) const {
    bool hit = false;
    double tEnd = INF;
    for (const auto &sphere: _spheres) {
        if (sphere.hit(ray, CLOSEST_POSSIBLE_RAY_HIT, tEnd, hitRes)) {
            hit = true;
            tEnd = hitRes.tOfHittingRay;
        }
    }
    return hit;
}

Color World::backgroundColor(const Ray &ray) {
    auto unitDir = unitVector(ray.direction());
    auto t = 0.5 * (unitDir.y() + 1.0);
    return alphaBlending({0.4, 0.4, 1}, {.9, .9, .9}, 1-t);
}


