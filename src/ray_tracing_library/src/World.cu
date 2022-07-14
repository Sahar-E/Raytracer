//
// Created by Sahar on 10/06/2022.
//

#include <string>
#include "my_math.h"
#include "commonCuda.cuh"
#include <Camera.cuh>
#include <vector>
#include <cfloat>
#include "HitResult.h"
#include "World.cuh"
#include "Vec3.cuh"
#include "commonDefines.h"


__host__ __device__ int World::getTotalSizeInMemoryForObjects() const {
    return getNSpheres() * sizeof(Sphere);
}

__host__ __device__
Color World::backgroundColor(const Ray &ray) {
    float intensity = 0.5f;
    auto unitDir = normalize(ray.direction());
    auto t = 0.5f * (unitDir.y() + 1.0f);
    return alphaBlending({0.4f, 0.4f, 1.0f}, {.9f, .9f, .9f}, t) * intensity;
}

__host__ __device__
int World::getNSpheres() const {
    return _nSpheres;
}

__host__ __device__
Sphere *World::getSpheres() const {
    return _spheres;
}


World World::initWorld2() {
    std::vector<Sphere> sphereList{};
    Color brown = {118 / 255.0, 91 / 255.0, 70 / 255.0};
    Material lambertianBrown = Material::getLambertian(brown);
    auto floor = Sphere({0, -5000, 0}, 5000, lambertianBrown);
    sphereList.push_back(floor);


    Color white = {1, 1, 1};
    Material whiteLight = Material::getGlowing(white * .8, white, 10);
    sphereList.push_back(Sphere({4, 5, 4}, 1, whiteLight));


    float radius = 1;
    std::vector<Vec3> bigBallsLocs = {{1.5,  radius, 4},
                                      {0.5,  radius, 2},
                                      {-0.5, radius, 0}};
    Material mirror = Material::getSpecular(white, white, 0.0, 1.0);
    sphereList.emplace_back(Sphere(bigBallsLocs[0], radius, mirror));
    Color redBrown = {141 / 255.0, 78 / 255.0, 44 / 255.0};
    Material lambertianRedBrown = Material::getLambertian(redBrown);
    sphereList.emplace_back(Sphere(bigBallsLocs[1], radius, lambertianRedBrown));
    Material glass = Material::getGlass(white, 1.5);
    sphereList.emplace_back(Sphere(bigBallsLocs[2], radius, glass));

    float smallSphereRadius = 0.2;
    int randState = 1;
    for (int xLoc = -8; xLoc < 8; ++xLoc) {
        for (int zLoc = -17; zLoc < 12; ++zLoc) {
            bool locationIsOccupied;
            for (auto & bigBallsLoc : bigBallsLocs) {
                locationIsOccupied = fabs(bigBallsLoc[0] - xLoc) < 1 && fabs(bigBallsLoc[2] - zLoc) < 1;
                if (locationIsOccupied) {
                    break;
                }
            }
            if (!locationIsOccupied) {
                Point3 sphereLoc = {xLoc + 0.7f * randomFloat(randState), smallSphereRadius, zLoc + 0.7f * randomFloat(
                        randState)};
                float randomMaterialChooser = randomFloat(randState);
                Material mat;
                if (randomMaterialChooser < 0.5) {
                    auto albedo = randomVec0to1(randState) * randomVec0to1(randState);
                    mat = Material::getLambertian(albedo);
                } else if (randomMaterialChooser < 0.8) {
                    auto albedo = randomVec0to1(randState) * randomVec0to1(randState);
                    auto specularColor = albedo + randomVec0to1(randState) * 0.2;
                    mat = Material::getSpecular(albedo, specularColor, randomFloat(randState), randomFloat(randState));
                } else if (randomMaterialChooser < 0.9) {
                    auto albedo = randomVec0to1(randState) * randomVec0to1(randState);
                    auto emittedColor = randomVec0to1(randState) * randomVec0to1(randState);
                    mat = Material::getGlowing(albedo, emittedColor, 8);
                } else {
                    mat = Material::getGlass(white, 1.5);
                }
                sphereList.emplace_back(Sphere(sphereLoc, smallSphereRadius, mat));
            }
        }
    }

    return {sphereList.data(), static_cast<int>(sphereList.size())};
}

World World::initWorld1() {
    std::vector<Sphere> sphereList{};
    Color red = {0.8, 0.2, 0.1};
    Color white = {1, 1, 1};
    Color gold1 = {212 / 255.0, 175 / 255.0, 55 / 255.0};
    Color gold2 = {218 / 255.0, 165 / 255.0, 32 / 255.0};
    Color bezh = {227 / 255.0, 203 / 255.0, 165 / 255.0};
    Color green = {51 / 255.0, 83 / 255.0, 69 / 255.0};
    Color brown = {118 / 255.0, 91 / 255.0, 70 / 255.0};
    Color redBrown = {141 / 255.0, 78 / 255.0, 44 / 255.0};
    Color darkGreen = {74 / 255.0, 71 / 255.0, 51 / 255.0};
    Color neonPurple = {176 / 255.0, 38 / 255.0, 255 / 255.0};
    Color neonGreen = {57 / 255.0, 255 / 255.0, 20 / 255.0};

//    Material metalGreen = Material(whiteGreenish, 0.99);
    Material mirror = Material::getSpecular(white, white, 0.0, 1.0);
    Material metalRedBrown = Material::getSpecular(redBrown, redBrown, 0.3, 0.3);
    Material metalGold = Material::getSpecular(gold1, gold2, 0.25, 1);
    Material lambertianBezh = Material::getLambertian(bezh);
    Material lambertianGreen = Material::getLambertian(green);
    Material lambertianBrown = Material::getLambertian(brown);
    Material lambertianRedBrown = Material::getLambertian(redBrown);
    Material metalGreen = Material::getSpecular(green, green, 0.2, 0.3);
    Material lambertianDarkGreen = Material::getLambertian(darkGreen);

    Material neonPurpleGlow = Material::getGlowing(white * .8, neonPurple, 4);
    Material neonGreenGlow = Material::getGlowing(white * .8, neonGreen, 4);
    Material whiteLight = Material::getGlowing(white * .8, white, 10);

    Material glass = Material::getGlass(white, 1.5);

    sphereList.push_back(Sphere({4, 4.5, -3}, 0.5, whiteLight));

    sphereList.push_back(Sphere({0, 0, -2}, 0.5, lambertianRedBrown));
    sphereList.push_back(Sphere({1.2, 0, -3}, 0.5, mirror));
    sphereList.push_back(Sphere({-1.2, -0.2, -2.5}, 0.3, metalGold));

    sphereList.push_back(Sphere({0.6, -0.4, -2.2}, 0.1, lambertianDarkGreen));
    sphereList.push_back(Sphere({0.4, -0.4, -1.5}, 0.1, lambertianBezh));
    sphereList.push_back(Sphere({0.5, -0.4, -1.2}, 0.1, glass));
    sphereList.push_back(Sphere({1, -0.4, -1.5}, 0.1, metalGreen));
    sphereList.push_back(Sphere({-0.9, -0.4, -3.3}, 0.1, lambertianGreen));
    sphereList.push_back(Sphere({-0.5, -0.4, -1.8}, 0.1, mirror));
    sphereList.push_back(Sphere({-0.8, -0.4, -1.7}, 0.1, neonPurpleGlow));
    sphereList.push_back(Sphere({-0.15, -0.4, -1.1}, 0.1, lambertianGreen));
    sphereList.push_back(Sphere({-0.45, -0.4, -1.2}, 0.1, glass));
    sphereList.push_back(Sphere({0.15, -0.4, -1.45}, 0.1, neonGreenGlow));
    sphereList.push_back(Sphere({-0.8, -0.4, -1.3}, 0.1, lambertianRedBrown));
    sphereList.push_back(Sphere({0, -1000.5, -2}, 1000, lambertianBrown));
    return {sphereList.data(), static_cast<int>(sphereList.size())};
}

__host__ __device__ World::World(const Sphere *spheresArr, int numSpheres)
        : _spheres(numSpheres <= 0 && spheresArr != nullptr ? nullptr : new Sphere[numSpheres]),
          _nSpheres(numSpheres) {
    for (int i = 0; i < _nSpheres; ++i) {
        _spheres[i] = spheresArr[i];
    }
}

__host__ __device__ World &World::operator=(World other) {
    swap(*this, other);
    return *this;
}

__host__ __device__ World::~World() {
    if (_spheres != nullptr) {
        delete[] _spheres;
        _spheres = nullptr;
    }
}

__host__ __device__ void World::swap(World &first, World &second) {
    auto temp1 = second._nSpheres;
    second._nSpheres = first._nSpheres;
    first._nSpheres = temp1;

    auto temp2 = second._spheres;
    second._spheres = first._spheres;
    first._spheres = temp2;
}

__host__ __device__ World::World(const World &other) : World(other.getSpheres(), other.getNSpheres()) {}
