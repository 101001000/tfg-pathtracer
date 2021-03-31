#ifndef SPHERE_H
#define SPHERE_H

#include "RenderableObject.h"
#include "Hit.h"
#include "Ray.h"
#include "Texture.h"

class Sphere : public RenderableObject {

public:
    float radius;

public:
    __host__ __device__ inline Sphere() {
        radius = 1;
    }

    __host__ __device__ inline Sphere(float _radius) {
        radius = _radius;
    }

    __host__ __device__ inline bool hit(Ray& ray, Hit& hit){

        Vector3 oc = ray.origin - position;

        float  a = ray.direction.length() * ray.direction.length();

        float half_b = Vector3::dot(oc, ray.direction);

        float c = oc.length() * oc.length() - radius * radius;

        float discriminant = half_b * half_b - a * c;
        if (discriminant < 0) return false;
        float sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        float root = (-half_b - sqrtd) / a;
        if (root < 0.01 || 1000000 < root) {
            root = (-half_b + sqrtd) / a;
            if (root < 0.01 || 1000000 < root)
                return false;
        }

        hit.t = root;
        hit.position = ray.origin + ray.direction * hit.t;
        hit.normal = ((hit.position - position) / radius).normalized();
        hit.objectID = objectID;
        hit.valid = true;
        hit.type = 0;
        hit.u = get_sphere_uv(hit.position).x;
        hit.v = get_sphere_uv(hit.position).y;

        return true;
    }


    __host__ __device__ Vector3 get_sphere_uv(const Vector3& _p) {

        float u, v;

        Texture::sphericalMapping(position, _p, radius, u, v);

        return Vector3(u, v, 0);
    }
};


#endif