#ifndef HDRI_H
#define HDRI_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Texture.h"
#include "Sampling.h"
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
		texture.data = new float[1024 * 1024 * 3];
		texture.width = 1024;
		texture.height = 1024;
		texture.USE_IMAGE = false;

		for (int i = 0; i < texture.width * texture.height; i++) {
			texture.data[3 * i + 0] = color.x;
			texture.data[3 * i + 1] = color.y;
			texture.data[3 * i + 2] = color.z;
		}

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

	inline void generateCDF2() {

		for (int y = 0; y < texture.height; y++) {
			for (int x = 0; x < texture.width; x++) {

				float r = 0;

				float u, v;

				Vector3 sample = UniformSampleSphere((float)x/(float)texture.width, (float)y/(float)texture.height);

				Texture::sphericalMapping(Vector3(), sample, 1, u, v);

				r += texture.getValueBilinear(u, v).x;
				r += texture.getValueBilinear(u, v).y;
				r += texture.getValueBilinear(u, v).z;

				radianceSum += r;
			}
		}	

		for (int y = 0; y < texture.height; y++) {
			for (int x = 0; x < texture.width; x++) {

				float r = 0;

				float u, v;

				Vector3 sample = UniformSampleSphere((float)x / (float)texture.width, (float)y / (float)texture.height);

				Texture::sphericalMapping(Vector3(), sample, 1, u, v);

				r += texture.getValueBilinear(u, v).x;
				r += texture.getValueBilinear(u, v).y;
				r += texture.getValueBilinear(u, v).z;

				r /= radianceSum;

				cdf[y*texture.width + x + 1] = r + cdf[y * texture.width + x];
			}
		}
	}

	inline void generateCDF() {

		radianceSum = 0;

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

			r /= radianceSum;

			cdf[i + 1] = r + cdf[i];
		}
	}

	int binarySearch(float* arr, float value, int length) {

		int from = 0;
		int to = length - 1;

		int m = (to - from) / 2;

		while (to - from > 1) {

			m = from + (to - from) / 2;

			if (to - from <= 0) return m;

			if (value < arr[m]) {
				to = m;
			}
			else {
				from = m;
			}
		}

		return m;
	}

	inline float pdf(int x, int y) {

		Vector3 dv = texture.getValue(x, y);

		//float theta = (float)y * PI / (float)texture.height;
		//float a = sin(theta);

		//return (dv.x + dv.y + dv.z) * (1.0 / (2.0 * PI)) / radianceSum ;

		return (((dv.x + dv.y + dv.z)/ radianceSum) * (texture.width * texture.height * (1.0 / (2.0 * PI))));
	}

	inline float brdf(int x, int y) {

		Vector3 dv = texture.getValue(x, y);

		return ((dv.x + dv.y + dv.z) / radianceSum * texture.width * texture.height) ;
	}

	inline Vector3 sample(float r1) {
		
		int count = binarySearch(cdf, r1, texture.width * texture.height);

		int wu = count % texture.width;
		int wv = count / texture.width;

		return Vector3(wu, wv, count);
	}

	inline Vector3 sample2(float r1) {

		int count = binarySearch(cdf, r1, texture.width * texture.height);

		int wu = count % texture.width;
		int wv = count / texture.width;

		float u, v;

		Vector3 sample = UniformSampleSphere((float)wu / (float)texture.width, (float)wv / (float)texture.height);

		Texture::sphericalMapping(Vector3(), sample, 1, u, v);


		return Vector3(u * texture.width, v * texture.height , count);
	}
};

#endif