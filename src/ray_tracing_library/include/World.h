//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include <Vec3.hpp>
#include <Ray.hpp>

class World {
public:
    [[nodiscard]] Color traceRay(const Ray &ray) const;
};



