//
// Created by Sahar on 02/07/2022.
//

#include "my_math.h"

double randomDouble(int &randState) {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::minstd_rand generator(1);
    return distribution(generator);
}

double randomDouble(int &randState, double from, double to) {
    return randomDouble(randState) * (from - to) + from;
}
