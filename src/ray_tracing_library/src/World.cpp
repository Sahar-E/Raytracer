//
// Created by Sahar on 10/06/2022.
//

#include <utils.h>
#include <constants.h>
#include "World.h"

Color World::traceRay(const Ray &ray) const {
    Color resColor{};
    double tHit = INF;
    for (const auto &sphere: _spheres) {
        sphere.hit(ray, 0.001, tHit, resColor, tHit);
    }
    if(!fcmp(tHit, INF)) {
        return resColor;
    }
    return backgroundColor(ray);
}

Color World::backgroundColor(const Ray &ray) const {
    auto unitDir = unit_vector(ray.direction());
    auto t = 0.5 * (unitDir.y() + 1.0);
    return alphaBlending({0.4, 0.4, 1}, {1, 1, 1}, t);
}


