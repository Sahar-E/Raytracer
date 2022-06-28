//
// Created by Sahar on 10/06/2022.
//

#include "HitResult.h"
#include "constants.h"
#include "World.cuh"
#include "Vec3.cuh"
#include "utils.cuh"


__host__ __device__
Color World::rayTrace(const Ray &ray, int bounce, int &randState) const{
//    rayTrace0 = emitted1 + atten1 * rayTrace1 = emitted1 + atten1(emitted2 + atten2 * rayTrace2)
    auto curRay(ray);
    auto *attenuationColors = new Color[bounce];
    auto *emittedColors = new Color[bounce];
    int hitCount = 0;
    while (hitCount < bounce - 1) { // when hitCount = bounce-1, will get black scatterColor.
        HitResult hitRes;
        Material material;
        bool hit = getHitResult(curRay, hitRes, material);
        if (hit) {
            Color emittedColor{}, attenuation{};
            material.getColorAndSecondaryRay(hitRes, randState, emittedColor, attenuation, curRay);
            attenuationColors[hitCount] = attenuation;
            emittedColors[hitCount] = emittedColor;
            ++hitCount;
        } else {
            attenuationColors[hitCount] = backgroundColor(ray);
            emittedColors[hitCount] = {0,0,0};
            ++hitCount;
            break;
        }
    }
    if (hitCount == bounce - 1) {
        // when hitCount = bounce-1, will get black scatterColor.
        attenuationColors[hitCount] = {0, 0, 0};
        emittedColors[hitCount] = {0, 0, 0};
    }

    Color res = {1,1,1};
    for(int i = hitCount-1; i >= 0; --i) {
        res = emittedColors[i] + attenuationColors[i] * res;
    }
    delete[] attenuationColors;
    delete[] emittedColors;
    return res;

//    if (bounce <= 0) {
//        Color black = {0, 0, 0};
//        return black;
//    }
//    HitResult hitRes;
//    Material material;
//    bool hit = getHitResult(ray, hitRes, material);
//
//    if (hit) {
//        Color emittedColor{}, attenuation{};
//        Ray secondaryRay{};
//        material.getColorAndSecondaryRay(hitRes, randState, emittedColor, attenuation, secondaryRay);
//        Color scatterColor = rayTrace(secondaryRay, bounce-1, randState) * attenuation;
//        return emittedColor + scatterColor;
//    } else {
//        return World::backgroundColor(ray);
//    }
}

__host__ __device__
bool World::getHitResult(const Ray &ray, HitResult &hitRes, Material &material) const {
    bool hit = false;
    double tEnd = INF;
    int hitSphereIdx = -1;
    for (int i = 0; i < _nSpheres; ++i) {
        if (_spheres[i].hit(ray, CLOSEST_POSSIBLE_RAY_HIT, tEnd, hitRes)) {
            hit = true;
            tEnd = hitRes.tOfHittingRay;
            hitSphereIdx = i;
        }
    }
    if (hit) {
        material = _spheres[hitSphereIdx].getMaterial();
    }
    return hit;
}

__host__ __device__
Color World::backgroundColor(const Ray &ray) {
    double intensity = 0.5;
    auto unitDir = normalize(ray.direction());
    auto t = 0.5 * (unitDir.y() + 1.0);
    return alphaBlending({0.4, 0.4, 1}, {.9, .9, .9}, t) * intensity;
}

size_t World::getNSpheres() const {
    return _nSpheres;
}

Sphere *World::getSpheres() const {
    return _spheres;
}

//size_t World::getNSpheres() const {
//    return ;
//}

//const Sphere *World::getSpheres() const {
//    return _spheres;
//}


__host__ __device__
ArrayOfObjects::ArrayOfObjects(const Sphere *spheresArr, size_t numSpheres) : _nSpheres(numSpheres) {
    _spheres = new Sphere[_nSpheres];
    for (int i = 0; i < _nSpheres; ++i) {
        _spheres[i] = spheresArr[i];
    }
}

__host__ __device__
ArrayOfObjects &ArrayOfObjects::operator=(ArrayOfObjects other) {
    swap(*this, other);
    return *this;
}



__host__ __device__
size_t ArrayOfObjects::getNSpheres() const {
    return _nSpheres;
}

__host__ __device__
void ArrayOfObjects::setNSpheres(size_t nSpheres) {
    _nSpheres = nSpheres;
}

__host__ __device__
Sphere *ArrayOfObjects::getSpheres() const {
    return _spheres;
}

__host__ __device__
void ArrayOfObjects::setSpheres(Sphere *spheres) {
    _spheres = spheres;
}

__host__ __device__
void ArrayOfObjects::swap(ArrayOfObjects &first, ArrayOfObjects &second)
{
    auto temp1 = second._nSpheres;
    second._nSpheres = first._nSpheres;
    first._nSpheres = temp1;

    auto temp2 = second._spheres;
    second._spheres = first._spheres;
    first._spheres = temp2;
}

__host__ __device__
ArrayOfObjects::~ArrayOfObjects() {
    delete[] _spheres;
}

__host__ __device__
ArrayOfObjects::ArrayOfObjects(const ArrayOfObjects &other) : _nSpheres(other._nSpheres) {
    _spheres = new Sphere[_nSpheres];
    for (int i = 0; i < _nSpheres; ++i) {
        _spheres[i] = other._spheres[i];
    }
}
