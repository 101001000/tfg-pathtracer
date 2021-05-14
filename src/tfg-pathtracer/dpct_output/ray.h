#ifndef RAY_H
#define RAY_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Vector.h"

class Ray {

public:
	Vector3 origin;
	Vector3 direction;

public:
	inline Ray(Vector3 _origin, Vector3 _direction) {
		origin = _origin;
		direction = _direction;
		direction.normalize();
	}

};



#endif