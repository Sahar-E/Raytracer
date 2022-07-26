//
// Created by Sahar on 10/06/2022.
//

#include "Camera.cuh"
#include <cuda_runtime_api.h>
#include "glm/glm.hpp"
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

Camera::Camera(Point3 lookFrom, Point3 lookAt, Vec3 vUp, float aspectRatio, float vFov,
               float aperture, float focusDist)
        : _origin(lookFrom), _aspectRatio(aspectRatio), _lensRadius(aperture / 2), _focusDist(focusDist), _vUp(vUp), _vFov(vFov){
    setFocusDistViewport(_vFov);
    setCameraInDir(normalize((lookFrom - lookAt)));
}

void Camera::setFocusDistViewport(float vFov) {
    auto vTheta = deg2rad(vFov);
    auto h = atan(vTheta / 2);
    auto viewportHeight = 2.0f * h;
    auto viewportWidth = _aspectRatio * viewportHeight;
    _focusDistTimesViewportWidth = _focusDist * viewportWidth;
    _focusDistTimesViewportHeight = _focusDist * viewportHeight;
}

void Camera::setFocusDist(float focusDist) {
    _focusDist = focusDist;
    setFocusDistViewport(_vFov);
    _horizontalVec = _focusDistTimesViewportWidth * _xVec;
    _verticalVec = _focusDistTimesViewportHeight * _yVec;
    _lowerLeftCorner = getLowerLeftCorner();
}

void Camera::setAperture(float aperture) {
    _lensRadius = aperture / 2;
}

float Camera::getAperture() const {
    return _lensRadius * 2;
}



float Camera::getFocusDist() const {
    return _focusDist;
}

void Camera::setVFov(float vFov) {
    _vFov = vFov;
    setFocusDistViewport(_vFov);
    _horizontalVec = _focusDistTimesViewportWidth * _xVec;
    _verticalVec = _focusDistTimesViewportHeight * _yVec;
    _lowerLeftCorner = getLowerLeftCorner();
}

float Camera::getVFov() const {
    return _vFov;
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

void Camera::setAspectRatio(float aspectRatio) {
    _aspectRatio = aspectRatio;
    setFocusDistViewport(_vFov);
    setCameraInDir(_zVec);
}
