//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include <Vec3.hpp>
#include <Ray.hpp>
#include <utility>
#include "Sphere.h"

class World {
public:

    explicit World() = default;

    void addSphere(const Sphere &s) {
        _spheres.push_back(s);
    }

    [[nodiscard]] Color traceRay(const Ray &ray) const;

    [[nodiscard]] static Color backgroundColor(const Ray &ray) ;

private:
    std::vector<Sphere> _spheres{};
};



