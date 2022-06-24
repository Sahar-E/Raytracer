//
// Created by Sahar on 10/06/2022.
//

#include "Camera.h"

Camera::Camera(Vec3 lookFrom, Vec3 lookAt, Vec3 vUp, double aspectRatio, double vFov)
        : _origin(lookFrom), _aspectRatio(aspectRatio) {
    auto vTheta = deg2rad(vFov);
    double focal_length = 1.0;
    auto h = tan(vTheta / 2); // Multiplied by focal_length
    auto viewportHeight = 2.0 * h;
    auto viewportWidth = _aspectRatio * viewportHeight;

    Vec3 zVec = normalize((lookFrom - lookAt));
    Vec3 xVec = normalize(cross(vUp, zVec));
    Vec3 yVec = cross(zVec, xVec);

    _horizontalVec = viewportWidth * xVec;
    _verticalVec = viewportHeight * yVec;
    _lowerLeftCorner = lookFrom - _horizontalVec / 2 - _verticalVec / 2 - zVec;
}

Ray Camera::getRay(double h_scalar, double v_scalar) const {
    return {_origin, _lowerLeftCorner + h_scalar*_horizontalVec + v_scalar*_verticalVec - _origin};
}

