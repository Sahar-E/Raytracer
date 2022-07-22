//
// Created by Sahar on 02/07/2022.
//

#pragma once

#include <random>
#include "glm/vec3.hpp"
#include "glm/glm.hpp"
#include <glm/gtx/quaternion.hpp>

float randomFloat(int &randState);
float randomFloat(int &randState, float from, float to);

glm::vec3 applyRotOnVec(const glm::quat &q, const glm::vec3 &vec);