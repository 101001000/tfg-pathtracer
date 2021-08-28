#ifndef MESHOBJECT_H
#define MESHOBJECT_H

#include "RenderableObject.hpp"
#include "Tri.hpp"
#include "Ray.hpp"
#include "Hit.hpp"

class MeshObject : public RenderableObject {

public:
	Tri* tris;
	unsigned int triCount;

public:


	void moveAbsolute(Vector3 newPosition) {

		Vector3 dif = newPosition - position;

		for (int i = 0; i < triCount; i++) {
			tris[i].vertices[0] += dif;
			tris[i].vertices[1] += dif;
			tris[i].vertices[2] += dif;
		}

		position = newPosition;
	}

	__host__ __device__ inline bool hit(Ray& ray, Hit& hit, bool shadowSmooth){

		Hit tempHit = Hit();

		for (int i = 0; i < triCount; i++) {

			if (tris[i].hit(ray, tempHit)) {

				if (!hit.valid) hit = tempHit;

				if((ray.origin - tempHit.position).length() < (ray.origin - hit.position).length()) hit = tempHit;
			}
		}

		if (hit.valid) {
			hit.objectID = objectID;
		}

		return hit.valid;
	}
};

#endif