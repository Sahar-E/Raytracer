//
// Created by Sahar on 10/06/2022.
//

#pragma once


#include "Vec3.cuh"
#include "Ray.cuh"

class Camera {

public:

    /**
     * Camera object that after initialization, can return rays going through the image plane using "lens" for
     * DOF (depth of field) affect.
     *
     * @param lookFrom      Origin of the camera.
     * @param lookAt        Target of the camera.
     * @param vUp           Vector of up direction in the world space.
     * @param aspectRatio   Aspect ratio of the image between the width and the height.
     * @param vFov          Vertical FOV of the image plane in degrees.
     * @param aperture      Bigger aperture means smaller DOF ("More blurred background").
     * @param focusDist     The distance of the focus plane from the origin of the camera.
     */
    Camera(Vec3 lookFrom, Vec3 lookAt, Vec3 vUp, float aspectRatio, float vFov, float aperture, float focusDist);

    /**
     * Returns ray using the specified scalars.
     * The scalars are in the range [0,1] where (0,0) is the lower left corner.
     *
     * @param h_scalar  horizontal scalar between [0,1].
     * @param v_scalar  vertical scalar between [0,1].
     * @param randState randState for generating random numbers.
     * @return  New ray in that direction from the camera origin.
     */
    __device__
    Ray getRay(float h_scalar, float v_scalar, curandState *randState) const;

private:
    Point3 _origin{};
    Point3 _lowerLeftCorner{};
    Vec3 _horizontalVec{};
    Vec3 _verticalVec{};

    Vec3 zVec{};
    Vec3 xVec{};
    Vec3 yVec{};

    float _aspectRatio{};
    float _lensRadius{};
};



