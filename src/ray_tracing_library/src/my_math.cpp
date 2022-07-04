//
// Created by Sahar on 02/07/2022.
//

#include "my_math.h"

float randomFloat(int &randState) {
    static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    static std::minstd_rand generator(1);
    return distribution(generator);
}

float randomFloat(int &randState, float from, float to) {
    return randomFloat(randState) * (from - to) + from;
}
