#ifndef HDRI_H
#define HDRI_H


#include "Texture.hpp"
#include "Sampling.hpp"
#include "HdrLoader.hpp"


class HDRI {

public:

	Texture texture;
	float* cdf;
	float radianceSum = 0;

	__host__ __device__ HDRI::HDRI() {
		texture = Texture();
	}

	__host__ __device__ HDRI::HDRI(Vector3 color) {

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

	HDRI::HDRI(std::string filepath) {

		int width, height;

		texture.data = loadHDR(filepath.c_str(), width, height);

		texture.width = width;
		texture.height = height;
		texture.USE_IMAGE = true;

		cdf = new float[texture.width * texture.height + 1];

		generateCDF();
	}

	__host__ __device__ inline void HDRI::generateCDF2() {

		for (int y = 0; y < texture.height; y++) {
			for (int x = 0; x < texture.width; x++) {

				float r = 0;
				float u, v;

				Vector3 sample = uniformSampleSphere((float)x / (float)texture.width, (float)y / (float)texture.height);

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

				Vector3 sample = uniformSampleSphere((float)x / (float)texture.width, (float)y / (float)texture.height);

				Texture::sphericalMapping(Vector3(), sample, 1, u, v);

				r += texture.getValueBilinear(u, v).x;
				r += texture.getValueBilinear(u, v).y;
				r += texture.getValueBilinear(u, v).z;

				r /= radianceSum;

				cdf[y * texture.width + x + 1] = r + cdf[y * texture.width + x];
			}
		}
	}

	__host__ __device__ inline void HDRI::generateCDF() {

		radianceSum = 0;

		cdf[0] = 0;

		for (int j = 0; j < texture.height; j++) {
			for (int i = 0; i < texture.width; i++) {
				Vector3 data = texture.getValueFromCoordinates(i, j);
				radianceSum += data.x + data.y + data.z;
			}
		}

		int c = 0;

		for (int j = 0; j < texture.height; j++) {
			for (int i = 0; i < texture.width; i++) {

				Vector3 data = texture.getValueFromCoordinates(i, j);
				//cdf[c + 1] = (data.x + data.y + data.z) / radianceSum + cdf[c];

				cdf[c + 1] = cdf[c] + (data.x + data.y + data.z) / radianceSum;
				c++;
			}
		}
	}

	__host__ __device__ int HDRI::binarySearch(float* arr, float value, int length) {

		int from = 0;
		int to = length - 1;

		while (to - from > 0) {

			int m = from + (to - from) / 2;

			if (value == arr[m]) return m;
			if (value < arr[m])	to = m - 1;
			if (value > arr[m]) from = m + 1;
		}

		return to;
	}


	__host__ __device__ float HDRI::pdf(int x, int y) {

		Vector3 dv = texture.getValueFromCoordinates(x, y);

		float theta = (((float)y / (float)texture.height)) * PI;

		// Semisphere area
		return ((dv.x + dv.y + dv.z) / radianceSum) * texture.width * texture.height / (2.0 * PI * PI * sin(theta));

	}

	__host__ __device__ Vector3 HDRI::sample(float r1) {

		int count = binarySearch(cdf, r1, texture.width * texture.height);

		int x = count % texture.width;
		int y = count / texture.width;

		//x = (int)(texture.xTile * (x + texture.xOffset * texture.width)) % texture.width;
		//y = (int)(texture.yTile * (y + texture.yOffset * texture.height)) % texture.height;

		return Vector3(x, y, count);
	}

	__host__ __device__ inline Vector3 HDRI::sample2(float r1) {

		int count = binarySearch(cdf, r1, texture.width * texture.height);

		int wu = count % texture.width;
		int wv = count / texture.width;

		float u, v;

		Vector3 sample = uniformSampleSphere((float)wu / (float)texture.width, (float)wv / (float)texture.height);

		Texture::sphericalMapping(Vector3(), sample, 1, u, v);

		return Vector3(u * texture.width, v * texture.height, count);
	}

};

#endif