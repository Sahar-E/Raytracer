//
// Created by Sahar on 10/06/2022.
//

#include "Camera.h"

Camera::Camera(Vec3 lookFrom, Vec3 lookAt, Vec3 vUp, double aspectRatio, double vFov,
               double aperture, double focusDist)
        : _origin(lookFrom), _aspectRatio(aspectRatio), _lensRadius(aperture / 2) {
    auto vTheta = deg2rad(vFov);
    auto h = tan(vTheta / 2);
    auto viewportHeight = 2.0 * h;
    auto viewportWidth = _aspectRatio * viewportHeight;

    zVec = normalize((lookFrom - lookAt));
    xVec = normalize(cross(vUp, zVec));
    yVec = cross(zVec, xVec);

    _horizontalVec = focusDist * viewportWidth * xVec;
    _verticalVec = focusDist * viewportHeight * yVec;
    _lowerLeftCorner = lookFrom - _horizontalVec / 2 - _verticalVec / 2 - focusDist * zVec;
}

Ray Camera::getRay(double h_scalar, double v_scalar) const {
    Vec3 random2dVec = randomVecInUnitDisk() * _lensRadius;
    Vec3 offset = xVec * random2dVec.x() + yVec * random2dVec.y();

    const Vec3 &rayOrigin = _origin + offset;
    const Vec3 &rayDirection = _lowerLeftCorner +
                               h_scalar * _horizontalVec +
                               v_scalar * _verticalVec - (_origin + offset);
    return {rayOrigin, rayDirection};
}