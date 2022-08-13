//
// Created by Sahar on 02/07/2022.
//

#include "my_math.h"

float randomFloat() {
    static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    static std::minstd_rand generator(1);
    return distribution(generator);
}

float randomFloat(float from, float to) {
    return randomFloat() * (from - to) + from;
}

glm::vec3 applyRotOnVec(const glm::quat &q, const glm::vec3 &vec) {
    return q * vec * inverse(q);
}
