#ifndef HDRI_H
#define HDRI_H

#include "Texture.h"
#include <vector>



class HDRI {

public:

	Texture texture;

	float* cdf;
	float radianceSum = 0;


	HDRI() {
		texture = Texture();
	}

	HDRI(Vector3 color) {
		texture.data = new float[3];
		texture.width = 1;
		texture.height = 1;
		texture.USE_IMAGE = false;
		texture.data[0] = color.x;
		texture.data[1] = color.y;
		texture.data[2] = color.z;
		cdf = new float[texture.width * texture.height + 1];

		generateCDF();
	}

	HDRI(const char* filepath) {
	
		int width, height;

		texture.data = loadHDR(filepath, width, height);
		texture.width = width;
		texture.height = height;
		texture.USE_IMAGE = true;
		cdf = new float[texture.width * texture.height + 1];	

		generateCDF();

	}

	inline void generateCDF() {

		cdf[0] = 0;

		for (int i = 0; i < texture.width * texture.height; i++) {

			float r = 0;

			r += texture.data[3 * i + 0];
			r += texture.data[3 * i + 1];
			r += texture.data[3 * i + 2];

			radianceSum += r;
		}

		for (int i = 0; i < texture.width * texture.height; i++) {

			float r = 0;

			r += texture.data[3 * i + 0];
			r += texture.data[3 * i + 1];
			r += texture.data[3 * i + 2];

			cdf[i + 1] = r + cdf[i];
		}
	}

	__device__ int binarySearch(float* arr, int value, int length) {

		int from = 0;
		int to = length - 1;

		int m = (to - from) / 2;

		while (to - from > 1) {

			m = from + (to - from) / 2;

			if (value >= arr[m] && value <= arr[m + 1])
				return m;

			if (value < arr[m]) {
				to = m;
			}
			else {
				from = m + 1;
			}
		}

		return m;
	}



	__device__ inline float pdf(int x, int y) {

		Vector3 dv = texture.getValue(x, y);

		return ((dv.x + dv.y + dv.z) / radianceSum) * (texture.width * texture.height * (1.0 / (2.0 * PI)));
	}

	__device__ inline Vector3 sample(float r1, float r2) {

		float v = 0;

		int count = binarySearch(cdf, r1*radianceSum, texture.width * texture.height);

		return Vector3(count%texture.width, (int)(count/texture.height), 0);
	}
};

#endif