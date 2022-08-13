//
// Created by Sahar on 02/07/2022.
//

#pragma once

#include <random>
#include "glm/vec3.hpp"
#include "glm/glm.hpp"
#include <glm/gtx/quaternion.hpp>

/**
 * @return  random number using a uniform distribution between 0 and 1.
 */
float randomFloat();

/**
 * random number using a uniform distribution between [from, to).
 * @param from  start range including.
 * @param to    end range excluding.
 * @return  the random number.
 */
float randomFloat(float from, float to);

/**
 * Apply rotation quaternion to a vector.
 * @param q     the rotation quaternion.
 * @param vec   the vector to apply the rotation to.
 * @return  the result.
 */
glm::vec3 applyRotOnVec(const glm::quat &q, const glm::vec3 &vec);