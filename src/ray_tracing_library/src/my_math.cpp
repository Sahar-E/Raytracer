//
// Created by Sahar on 02/07/2022.
//

#include "my_math.h"

float randomDouble(int &randState) {
    static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    static std::minstd_rand generator(1);
    return distribution(generator);
}

float randomDouble(int &randState, float from, float to) {
    return randomDouble(randState) * (from - to) + from;
}
