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
    Camera(Point3 lookFrom, Point3 lookAt, Vec3 vUp, float aspectRatio, float vFov, float aperture, float focusDist);

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


    /**
     * Rotates the camera by the specified angles.
     * @param hRot     Yaw rotation in degrees.
     * @param vRot     Pitch rotation in degrees.
     */
    void rotateCamera(float hRot, float vRot);

    /**
     * Camera movement function for the WASD keys. Moves the camera along the xAxis and zAxis.
     * @param forward   Amount to move the camera forward (negative will move back).
     * @param right     Amount to move the camera right (negative will move left).
     */
    void moveCameraForward(float forward);

    /**
     * Camera movement function for the WASD keys. Moves the camera along the xAxis and zAxis.
     * @param right     Amount to move the camera right (negative will move left).
     */
    void moveCameraRight(float right);

    /**
     * Camera movement function for the WASD keys. Moves the camera along the xAxis and zAxis.
     * @param forward   Amount to move the camera forward (negative will move back).
     * @param right     Amount to move the camera right (negative will move left).
     */
    void moveCameraUp(float up);

    void setCameraInDir(const Vec3 &zVec);

    Vec3 getLowerLeftCorner();

    void setFocusDistViewport(float vFov);

    float getVFov() const;
    void setVFov(float vFov);
    float getFocusDist() const;
    void setFocusDist(float focusDist);
    float getAperture() const;
    void setAperture(float focusDist);

private:
    Point3 _origin{};
    Point3 _lowerLeftCorner{};
    Vec3 _horizontalVec{};
    Vec3 _verticalVec{};

    Vec3 _zVec{};
    Vec3 _xVec{};
    Vec3 _yVec{};

    float _aspectRatio{};
    float _lensRadius{};
    float _focusDistTimesViewportWidth;
    float _focusDistTimesViewportHeight;
    float _focusDist;
    Vec3 _vUp;
    float _vFov{};

};



