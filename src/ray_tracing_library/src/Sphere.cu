//
// Created by Sahar on 10/06/2022.
//

#include "Sphere.cuh"

__host__ __device__
void Sphere::getHitResult(const Ray &ray, double rootRes, HitResult &hitRes) const {
    Point3 hitPoint = ray.at(rootRes);
    Vec3 normalOfHitPoint = normalize(hitPoint - _center);
    bool isOutwards = dot(ray.direction(), normalOfHitPoint) < 0;
    hitRes = HitResult(ray,
                       isOutwards ? normalOfHitPoint : -normalOfHitPoint,
                       hitPoint,
                       rootRes,
                       isOutwards);
}

__host__ __device__
bool Sphere::isHit(const Ray &ray, double tStart, double tEnd, double &rootRes) const {
    Vec3 oc = ray.origin() - _center;
    auto a = ray.direction().length_squared();
    auto half_b = dot(oc, ray.direction());
    auto c = oc.length_squared() - _radius*_radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) {
        return false;
    }
    auto sqrtD = std::sqrt(discriminant);
    auto root = (-half_b - sqrtD) / a;      // Closer root
    if (root < tStart || root > tEnd) {
        root = (-half_b + sqrtD) / a;              // Farther root
        if (root < tStart || root > tEnd) {
            return false;
        }
    }
    rootRes = root;
    return true;
}

__host__ __device__ const Material &Sphere::getMaterial() const {
    return _material;
}
