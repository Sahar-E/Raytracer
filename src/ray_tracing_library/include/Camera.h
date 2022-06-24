//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include <Vec3.h>
#include <Ray.hpp>

class Camera {

public:

    /**
     * Camera object that after initialization, can return rays going thourgh the image plane.
     *
     * @param lookFrom      Origin of the camera.
     * @param lookAt        Target of the camera.
     * @param vUp           Vector of up direction in the world space.
     * @param aspectRatio   Aspect ratio of the image between the width and the height.
     * @param vFov          Vertical FOV of the image plane in degrees.
     */
    Camera(Vec3 lookFrom, Vec3 lookAt, Vec3 vUp, double aspectRatio, double vFov);

    /**
     * Returns ray using the specified scalars.
     * The scalars are in the range [0,1] where (0,0) is the lower left corner.
     *
     * @param h_scalar  horizontal scalar between [0,1].
     * @param v_scalar  vertical scalar between [0,1].
     * @return  New ray in that direction from the camera origin.
     */
    [[nodiscard]] Ray getRay(double h_scalar, double v_scalar) const;

private:
    Point3 _origin;
    Point3 _lowerLeftCorner;
    Vec3 _horizontalVec;
    Vec3 _verticalVec;
    double _aspectRatio;
};



