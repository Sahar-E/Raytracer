//
// Created by Sahar on 10/06/2022.
//

#include "Camera.h"

Camera::Camera(Vec3 origin, double aspectRatio, double vFov) : _origin(origin), _aspectRatio(aspectRatio){
    auto vTheta = deg2rad(vFov);
    double focal_length = 1.0;
    auto h = tan(vTheta / 2); // Multiplied by focal_length
    auto viewportHeight = 2.0 * h;
    auto viewportWidth = _aspectRatio * viewportHeight;


    _horizontalVec = Vec3(viewportWidth, 0, 0);
    _verticalVec = Vec3(0, viewportHeight, 0);
    _lowerLeftCorner = origin - _horizontalVec/2 - _verticalVec/2 - Vec3(0, 0, focal_length);
}

Ray Camera::getRay(double h_scalar, double v_scalar) const {
    return {_origin, _lowerLeftCorner + h_scalar*_horizontalVec + v_scalar*_verticalVec - _origin};
}

