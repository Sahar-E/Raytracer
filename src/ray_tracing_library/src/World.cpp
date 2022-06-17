//
// Created by Sahar on 10/06/2022.
//

#include <HitResult.h>
#include <constants.h>
#include <utils.h>
#include "World.h"

Color World::rayTrace(const Ray &ray, int bounce) const {
    if (bounce <= 0) {
        const static Color black = {0, 0, 0};
        return black;
    }
    HitResult hitRes;
    bool hit = false;
    double tEnd = INF;
    for (const auto &sphere: _spheres) {
        if (sphere.hit(ray, CLOSEST_POSSIBLE_RAY_HIT, tEnd, hitRes)) {
            hit = true;
            hitRes.color = hitRes.color;
            tEnd = hitRes.tOfHittingRay;
        }
    }

    if (hit) {
        Vec3 rayRandomDirection = randomVecOnTangentSphere(hitRes.normal, hitRes.hitPoint) - hitRes.hitPoint;
        Ray diffusedRay{hitRes.hitPoint, rayRandomDirection};
        return rayTrace(diffusedRay, bounce - 1) * 0.5;

//        const Ray &reflectionRay = hitRes.reflectionRay;
//        return rayTrace(reflectionRay, bounce - 1) * 0.5;
    } else {
        return World::backgroundColor(ray);
    }
}

Color World::backgroundColor(const Ray &ray) {
    auto unitDir = unitVector(ray.direction());
    auto t = 0.5 * (unitDir.y() + 1.0);
    return alphaBlending({0.4, 0.4, 1}, {1, 1, 1}, t);
}


