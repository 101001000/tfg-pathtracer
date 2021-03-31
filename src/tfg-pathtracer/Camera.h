#ifndef CAMERA_H
#define CAMERA_H

#include "SceneObject.h"

class Camera : public SceneObject {

public:
	unsigned int xRes, yRes;

	float focalLength; //mm
	float sensorWidth; //mm
	float sensorHeight; //mm

public:

	//TODO: poner sensorWidth de manera elegante

	__host__ __device__ Camera() {
		xRes = 1280;
		yRes = 720;
		focalLength = 35;
		sensorWidth = 35;
		sensorHeight = sensorWidth * ((float)yRes / (float)xRes);
	}

	__host__ __device__ Camera(unsigned int _xRes, unsigned int _yRes) {
		xRes = _xRes;
		yRes = _yRes;
		focalLength = 35;
		sensorWidth = 35;
		sensorHeight = sensorWidth * ((float)yRes / (float)xRes);
	}




};


#endif
