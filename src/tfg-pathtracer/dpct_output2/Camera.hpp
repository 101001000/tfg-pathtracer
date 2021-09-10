#ifndef CAMERA_H
#define CAMERA_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Vector.hpp"

class Camera {

public:

	unsigned int xRes = 1280;
	unsigned int yRes = 720;

	float focalLength = 35 * 0.001;
	float sensorWidth = 35 * 0.001;
	float sensorHeight = sensorWidth * ((float)yRes / (float)xRes);
	float aperture = 2.8;
	float focusDistance = 1000000;

	Vector3 position = Vector3::Zero();

public:

	Camera() {}

	Camera(unsigned int _xRes, unsigned int _yRes) {
		xRes = _xRes;
		yRes = _yRes;
		setSensorWidth(sensorWidth);
	}

	void setSensorWidth(float size) {
		sensorHeight = size * ((float)yRes / (float)xRes);
                /*
                DPCT1040:1: Use sycl::stream instead of printf, if your code is
                used on the device.
                */
                printf("sensorWidth %f sensorHeight %f", sensorWidth, sensorHeight);
        }


};


#endif
