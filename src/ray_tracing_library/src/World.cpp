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
    Material material;
    bool hit = getHitResult(ray, hitRes, material);

    if (hit) {
        auto [emittedColor, attenuation, secondaryRay] = material.getColorAndSecondaryRay(hitRes);
        Color scatterColor = rayTrace(secondaryRay, bounce - 1) * attenuation;
        return emittedColor + scatterColor;
    } else {
        return World::backgroundColor(ray);
    }
}

bool World::getHitResult(const Ray &ray, HitResult &hitRes, Material &material) const {
    bool hit = false;
    double tEnd = INF;
    int hitSphereIdx = -1;
    for (int i = 0; i < _spheres.size(); ++i) {
        if (_spheres[i].hit(ray, CLOSEST_POSSIBLE_RAY_HIT, tEnd, hitRes)) {
            hit = true;
            tEnd = hitRes.tOfHittingRay;
            hitSphereIdx = i;
        }
    }
    if (hit) {
        material = _spheres[hitSphereIdx].getMaterial();
    }
    return hit;
}

Color World::backgroundColor(const Ray &ray) {
    auto unitDir = normalize(ray.direction());
    auto t = 0.5 * (unitDir.y() + 1.0);
    return alphaBlending({0.4, 0.4, 1}, {.9, .9, .9}, t) * 0.8;
}


