//
// Created by Sahar on 10/06/2022.
//

#include "Camera.cuh"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "my_math.h"
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <stdio.h>

Camera::Camera(Point3 lookFrom, Point3 lookAt, Vec3 vUp, float aspectRatio, float vFov,
               float aperture, float focusDist)
        : _origin(lookFrom), _aspectRatio(aspectRatio), _lensRadius(aperture / 2), _focusDist(focusDist), _vUp(vUp){
    auto vTheta = deg2rad(vFov);
    auto h = atan(vTheta / 2);
    auto viewportHeight = 2.0f * h;
    auto viewportWidth = _aspectRatio * viewportHeight;
    _focusDistTimesViewportWidth = _focusDist * viewportWidth;
    _focusDistTimesViewportHeight = _focusDist * viewportHeight;

    setCameraInDir(normalize((lookFrom - lookAt)));

    printf("%f %f %f\n", _zVec.x(), _zVec.y(), _zVec.z());
    glm::vec3 eye(lookFrom[0], lookFrom[1], lookFrom[2]);
    glm::vec3 center(lookAt[0], lookAt[1], lookAt[2]);
    auto mat = glm::lookAt(eye, center, {0, 1, 0});

    glm::mat4x4 view;
    view[0][0] = _xVec.x();
    view[1][0] = _xVec.y();
    view[2][0] = _xVec.z();
    view[3][0] = dot(-_xVec, _origin );
    view[0][1] = _yVec.x();
    view[1][1] = _yVec.y();
    view[2][1] = _yVec.z();
    view[3][1] = dot(-_yVec, _origin );
    view[0][2] = _zVec.x();
    view[1][2] = _zVec.y();
    view[2][2] = _zVec.z();
    view[3][2] = dot(-_zVec, _origin );
    view[0][3] = 0;
    view[1][3] = 0;
    view[2][3] = 0;
    view[3][3] = 1.0f;

//    glm::extractEulerAngleXYZ()


}

void Camera::setCameraInDir(const Vec3 &zVec) {
    _zVec = normalize(zVec);
    _xVec = normalize(cross(_vUp, _zVec));
    _yVec = cross(_zVec, _xVec);


    _horizontalVec = _focusDistTimesViewportWidth * _xVec;
    _verticalVec = _focusDistTimesViewportHeight * _yVec;
    _lowerLeftCorner = getLowerLeftCorner();
}

Vec3 Camera::getLowerLeftCorner() { return _origin - _horizontalVec / 2 - _verticalVec / 2 - _focusDist * _zVec; }

__device__ Ray Camera::getRay(float h_scalar, float v_scalar, curandState *randState) const {
    Vec3 random2dVec = randomVecInUnitDisk(randState) * _lensRadius;
    Vec3 offset = _xVec * random2dVec.x() + _yVec * random2dVec.y();

    const Vec3 &rayOrigin = _origin + offset;
    const Vec3 &rayDirection = _lowerLeftCorner +
                               h_scalar * _horizontalVec +
                               v_scalar * _verticalVec - (_origin + offset);
    return {rayOrigin, rayDirection};
}
// Using the UVN Vectors camera idea where in camera coordinates, the U is up vector, V is right vector and N is
// a vector pointing forward.

//    glm::vec3 vUp{_vUp[0], _vUp[1], _vUp[2]};
//    glm::vec3 V{1.0f, 0.0f, 0.0f};
//    auto horizontalRot = glm::angleAxis(h_rot, V);
//    V = applyRotOnVec(horizontalRot, V);
//
//    printf("%f\n", glm::l2Norm(V));
//
//    glm::vec3 U = glm::normalize(glm::cross(vUp, V));
//    auto verticalRot = glm::angleAxis(v_rot, U);
//    U = applyRotOnVec(verticalRot, U);
//
//    glm::vec3 N = glm::normalize(glm::cross(U, V));
//
//    glm::mat4x4 view;
//    view[0][0] = _xVec.x();
//    view[1][0] = _xVec.y();
//    view[2][0] = _xVec.z();
//    view[3][0] = dot(-_xVec, _origin );
//    view[0][1] = _yVec.x();
//    view[1][1] = _yVec.y();
//    view[2][1] = _yVec.z();
//    view[3][1] = dot(-_yVec, _origin );
//    view[0][2] = _zVec.x();
//    view[1][2] = _zVec.y();
//    view[2][2] = _zVec.z();
//    view[3][2] = dot(-_zVec, _origin );
//    view[0][3] = 0;
//    view[1][3] = 0;
//    view[2][3] = 0;
//    view[3][3] = 1.0f;
//    glm::vec4 z{_zVec[0], _zVec[1], _zVec[2], 0};
//    auto mat = glm::inverse(view) * z;

void Camera::rotateCamera(float hRot, float vRot) {
    glm::vec3 up{_vUp[0], _vUp[1], _vUp[2]};
    glm::vec3 target{_zVec[0], _zVec[1], _zVec[2]};
    target = glm::rotateY(target, deg2rad(hRot));
    glm::vec3 rotateAround = glm::cross(target, up);
    target = glm::normalize(glm::mat3(glm::rotate(deg2rad(vRot), rotateAround)) * target);
    setCameraInDir({target.x, target.y, target.z});
}

void Camera::moveCameraForward(float forward) {
    _origin += -_zVec * forward;
    _lowerLeftCorner = getLowerLeftCorner();    // origin changed, so we need to update the lower left corner.
}

void Camera::moveCameraRight(float right) {
    _origin += _xVec * right;
    _lowerLeftCorner = getLowerLeftCorner();    // origin changed, so we need to update the lower left corner.
}

void Camera::moveCameraUp(float up) {
    _origin += _yVec * up;
    _lowerLeftCorner = getLowerLeftCorner();    // origin changed, so we need to update the lower left corner.
}
