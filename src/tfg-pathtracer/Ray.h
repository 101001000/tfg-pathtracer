#ifndef RAY_H
#define RAY_H

#include "Vector.h"

class Ray {

public:
	Vector3 origin;
	Vector3 direction;

public:

	__host__ __device__ inline Ray(Vector3 _origin, Vector3 _direction) {
		origin = _origin;
		direction = _direction;
		direction.normalize();
	}

	__host__ __device__ inline Ray() {
		origin = Vector3();
		direction = Vector3(0,0,1);
	}

};



#endif