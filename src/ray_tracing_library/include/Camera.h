//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include <Vec3.hpp>
#include <Ray.hpp>

class Camera {

public:

    Camera(Vec3 origin);

    [[nodiscard]] Ray getRay(double h_scalar, double v_scalar) const;
private:
    Point3 _origin;
    Point3 _lowerLeftCorner;
    Vec3 _horizontalVec;
    Vec3 _verticalVec;
};



