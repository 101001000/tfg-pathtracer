#ifndef CAMERA_H
#define CAMERA_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "SceneObject.h"

class Camera : public SceneObject {

public:
	unsigned int xRes, yRes;

	float focalLength; //mm
	float sensorWidth; //mm
	float sensorHeight; //mm
	float aperture; // fStops
	float focusDistance = 1000000; //inf m

public:

	//TODO: poner sensorWidth de manera elegante

	Camera() {
		xRes = 1280;
		yRes = 720;
		focalLength = 35;
		sensorWidth = 35;
		sensorHeight = sensorWidth * ((float)yRes / (float)xRes);
		aperture = 2.8;
	}

	Camera(unsigned int _xRes, unsigned int _yRes) {
		xRes = _xRes;
		yRes = _yRes;
		focalLength = 35;
		sensorWidth = 35;
		sensorHeight = sensorWidth * ((float)yRes / (float)xRes);
		aperture = 2.8;
	}




};


#endif
