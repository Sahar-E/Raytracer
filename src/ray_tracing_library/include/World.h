//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include <vector>
#include "Sphere.h"

static const double CLOSEST_POSSIBLE_RAY_HIT = 0.001;

class World {
public:

    explicit World() = default;

    void addSphere(const Sphere &s) {
        _spheres.push_back(s);
    }

    [[nodiscard]] Color rayTrace(const Ray &ray, int bounce) const;

    [[nodiscard]] static Color backgroundColor(const Ray &ray) ;

private:
    std::vector<Sphere> _spheres{};
};



