#ifndef TRI_H
#define TRI_H
#include "MeshObject.hpp"

#define SMOOTH_SHADING true

class Tri {
public:

	Vector3 vertices[3];
    Vector3 uv[3];
    Vector3 normals[3];

    int objectID;

public:

    Tri() {

    }

    __host__ __device__ inline Vector3 centroid() {

        return Vector3(vertices[0].x + vertices[1].x + vertices[2].x,
            vertices[0].y + vertices[1].y + vertices[2].y,
            vertices[0].z + vertices[1].z + vertices[2].z) / 3.0;
    }

    __host__ __device__ inline Vector3 projectOnPlane(Vector3 position, Vector3 origin, Vector3 normal){
        return position - Vector3::dot(position - origin, normal) * normal;
    }

    __host__ __device__ inline bool hit(Ray& ray, Hit& hit, Vector3 position) {

        float EPSILON = 0.00001;

        Vector3 edge1 = vertices[1] - vertices[0];
        Vector3 edge2 = vertices[2] - vertices[0];

        Vector3 pvec = Vector3::cross(ray.direction, edge2);
        Vector3 N = Vector3::cross(edge1, edge2).normalized();

        float u, v, t, inv_det;

        float det = Vector3::dot(edge1, pvec);

        inv_det = 1.0 / det;

        if (det > -EPSILON && det < EPSILON) return false;

        Vector3 tvec = ray.origin - vertices[0];

        u = Vector3::dot(tvec, pvec) * inv_det;
        if (u < 0.0 || u > 1.0)
            return false;

        Vector3 qvec = Vector3::cross(tvec, edge1);
        v = Vector3::dot(ray.direction, qvec) * inv_det;
        if (v < 0.0 || (u + v) > 1.0)
            return false;

        t = Vector3::dot(edge2, qvec) * inv_det;

        if (t < 0) return false;

        // UV coordinates from the texture, weighted with vertex texture UV
        Vector3 tUV = uv[0] + (uv[1] - uv[0]) * u + (uv[2] - uv[0]) * v;
        Vector3 geomPosition = ray.origin + ray.direction * t;

#if SMOOTH_SHADING 

        // https://gist.github.com/pixnblox/5e64b0724c186313bc7b6ce096b08820

        Vector3 p0 = projectOnPlane(geomPosition, vertices[0], normals[0]);
        Vector3 p1 = projectOnPlane(geomPosition, vertices[1], normals[1]);
        Vector3 p2 = projectOnPlane(geomPosition, vertices[2], normals[2]);

        hit.normal = normals[0] + (normals[1] - normals[0]) * u + (normals[2] - normals[0]) * v;
        hit.position = p0 + (p1 - p0) * u + (p2 - p0) * v;
#else

        hit.normal = N;
        hit.position = geomPosition;
#endif

        hit.t = t;
        hit.valid = true;
        hit.type = 1;
        hit.u = tUV.x;
        hit.v = tUV.y;
        hit.objectID = objectID;

        return true;
    }
};

__host__ __device__ inline bool operator==(const Tri& t1, const Tri& t2) {
    return (t1.uv[0] == t2.uv[0]) &&
        (t1.uv[1] == t2.uv[1]) &&
        (t1.uv[2] == t2.uv[2]) &&
        (t1.vertices[0] == t2.vertices[0]) &&
        (t1.vertices[1] == t2.vertices[1]) &&
        (t1.vertices[2] == t2.vertices[2]);
}


#endif
