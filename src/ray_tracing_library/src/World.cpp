//
// Created by Sahar on 10/06/2022.
//

#include <utils.h>
#include "World.h"

Color World::traceRay(const Ray &ray) const {
    auto unitDir = unit_vector(ray.direction());
    auto t = 0.5 * (unitDir.y() + 1.0);
    return alphaBlending(Color(SKY_COLOR), Color(WHITE_COLOR), t);
}
