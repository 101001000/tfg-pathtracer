#ifndef MESHOBJECT_H
#define MESHOBJECT_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Tri.hpp"
#include "Ray.hpp"
#include "Hit.hpp"

static int objectIDCount = 0;

// @todo Define constructor for 


class MeshObject {

public:
	Tri* tris;

	unsigned int triCount;

	int materialID = 0;
	int objectID = 0;

public:

	MeshObject() {
		objectID = objectIDCount++;
	}

	void translate(Vector3 pos) {

		for (int i = 0; i < triCount; i++) {
			tris[i].vertices[0] += pos;
			tris[i].vertices[1] += pos;
			tris[i].vertices[2] += pos;
		}

	}

	inline bool hit(Ray& ray, Hit& hit, bool shadowSmooth){

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