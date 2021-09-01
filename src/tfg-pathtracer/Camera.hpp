#ifndef CAMERA_H
#define CAMERA_H

#include "Vector.hpp"

class Camera {

public:
	unsigned int xRes, yRes;

	Vector3 position;

	float focalLength;
	float sensorWidth;
	float sensorheight;
	float aperture;
	float focusDistance = 1000000;

public:

	__host__ __device__ Camera() {
		xRes = 1280;
		yRes = 720;
		focalLength = 35 * 0.001;
		sensorWidth = 36 * 0.001;
		sensorheight = sensorWidth * ((float)yRes / (float)xRes);
		aperture = 2.8;
	}

	__host__ __device__ Camera(unsigned int _xRes, unsigned int _yRes) {
		xRes = _xRes;
		yRes = _yRes;
		focalLength = 35 * 0.001;
		sensorWidth = 36 * 0.001;
		sensorheight = sensorWidth * ((float)yRes / (float)xRes);
		aperture = 2.8;
	}


};


#endif
