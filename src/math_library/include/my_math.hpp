//
// Created by Sahar on 08/06/2022.
//

#pragma once

#include <random>

/**
 * @return a random real in [0,1).
 */
inline double randomDouble() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::minstd_rand generator(1);
    return distribution(generator);
}

inline double randomDouble(double from, double to) {
    return randomDouble() * (from - to) + from;
}

inline double deg2rad(double degree) {
    return degree * M_PI / 180.0;
}