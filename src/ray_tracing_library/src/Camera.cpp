//
// Created by Sahar on 10/06/2022.
//

#include "Camera.h"

Camera::Camera(Vec3 origin, double widthToHeight) : _origin(origin), _widthToHeight(widthToHeight){
    double focal_length = 1.0;
    double viewPortHeight = 1.0;
    double viewPortWidth = _widthToHeight * viewPortHeight;

    _horizontalVec = Vec3(viewPortWidth, 0, 0);
    _verticalVec = Vec3(0, viewPortHeight, 0);
    _lowerLeftCorner = origin - _horizontalVec/2 - _verticalVec/2 - Vec3(0, 0, focal_length);
}

Ray Camera::getRay(double h_scalar, double v_scalar) const {
    return {_origin, _lowerLeftCorner + h_scalar*_horizontalVec + v_scalar*_verticalVec - _origin};
}

