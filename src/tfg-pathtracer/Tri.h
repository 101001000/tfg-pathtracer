#ifndef TRI_H
#define TRI_H
#include "MeshObject.h"

class MeshObject;


class Tri {
public:

	Vector3 vertices[3];
    Vector3 uv[3];

    int objectID;

public:
	Tri(Vector3 _p1, Vector3 _p2, Vector3 _p3) {
        vertices[0] = _p1;
        vertices[1] = _p2;
        vertices[2] = _p3;
	}

    Tri() {
        vertices[0] = Vector3();
        vertices[1] = Vector3();
        vertices[2] = Vector3();
    }

    __host__ __device__ inline Vector3 centroid() {

        return (1.0/3.0) * Vector3(vertices[0].x + vertices[1].x + vertices[2].x, vertices[0].y + vertices[1].y + vertices[2].y, vertices[0].z + vertices[1].z + vertices[2].z);

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

        if (0) {
            if (det < EPSILON)
                return false;

            Vector3 tvec = ray.origin - vertices[0];

            u = Vector3::dot(tvec, pvec);

            if (u < 0.0 || u > det)
                return false;

            Vector3 qvec = Vector3::cross(tvec, edge1);

            v = Vector3::dot(ray.direction, qvec);

            if (v < 0.0 || u + v > det)
                return false;

            t = Vector3::dot(edge2, qvec);

            t *= inv_det;
            u *= inv_det;
            v *= inv_det;
        }
        else {
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
        }

        if (t < 0) return false;

        // Operar aquí con vectores es ineficiente, mejor serparar u y v
        Vector3 nuv = uv[0] + (uv[1] - uv[0]) * u + (uv[2] - uv[0]) * v;

        hit.t = t;
        hit.position = ray.origin + ray.direction * hit.t;
        hit.normal = N;
        hit.valid = true;
        hit.type = 1;
        hit.u = nuv.x;
        hit.v = nuv.y;
        hit.objectID = objectID;

        return true;
    }
};

__host__ __device__ inline bool operator==(const Tri& t1, const Tri& t2) {

    return (t1.uv[0] == t2.uv[0]) && (t1.uv[1] == t2.uv[1]) && (t1.uv[2] == t2.uv[2]) && (t1.vertices[0] == t2.vertices[0]) && (t1.vertices[1] == t2.vertices[1]) && (t1.vertices[2] == t2.vertices[2]);
}


#endif
