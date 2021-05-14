#ifndef VECTOR_H
#define VECTOR_H

#include <cmath>
#include <string>
#include "cuda_runtime.h"

class Vector3 {

public:
	float x, y, z;

public:

	__host__ inline Vector3(std::string str) {
	
		//SE PUEDE SIMPLIFICAR CON UN �NDICE

		bool start = false;

		bool px = true;
		bool py = false;
		bool pz = false;

		std::string sx = "";
		std::string sy = "";
		std::string sz = "";

		for (char& c : str) {

			if (isdigit(c) || c == '.' || c == '-') {

				if (px) sx.push_back(c);
				if (py) sy.push_back(c);
				if (pz) sz.push_back(c);

				start = true;
			}
			else {
				if (start) {

					if (px) {
						px = false;
						py = true;
					} else if (py) {
						py = false;
						pz = true;
					} else if (pz) {
						pz = false;
					}
				}
			}
		}

		if (sx.size() == 0) sx.push_back('0');
		if (sy.size() == 0) sy.push_back('0');
		if (sz.size() == 0) sz.push_back('0');

		x = std::stof(sx);
		y = std::stof(sy);
		z = std::stof(sz);


	}

	__host__ __device__ inline Vector3() {
		x = 0;
		y = 0;
		z = 0;
	}

	__host__ __device__ inline Vector3(float _x, float _y, float _z) {
		x = _x;
		y = _y;
		z = _z;
	}

	__host__ __device__ inline Vector3(float _x) {
		x = _x;
		y = _x;
		z = _x;
	}

	__host__ __device__ inline float length() const {
		return sqrt(x * x + y * y + z * z);
	}

	__host__ __device__ inline Vector3 operator*(const float s) {
		return Vector3(x * s, y * s, z * s);
	}

	__host__ __device__ inline float operator[](const int n) {
		if (n == 0) return x;
		if (n == 1) return y;
		if (n == 2) return z;
		return x;
	}

	__host__ __device__ inline Vector3 operator/(const float s) {
		return Vector3(x / s, y / s, z / s);
	}

	__host__ __device__ inline bool operator==(const Vector3& v) {

		float EPSILON = 0.0001;

		return abs(x - v.x) < EPSILON && abs(y - v.y) < EPSILON && abs(z - v.z) < EPSILON;
	}

	__host__ __device__ inline bool operator!=(const Vector3& v) {
		return !(*this == v);
	}

	__host__ __device__ inline Vector3& operator+=(const Vector3& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	__host__ __device__ inline Vector3& operator-=(const Vector3& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	__host__ __device__ inline Vector3& operator*=(const Vector3& v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
		return *this;
	}

	__host__ __device__ inline Vector3& operator*=(float s) {
		x *= s;
		y *= s;
		z *= s;
		return *this;
	}

	__host__ __device__ inline static Vector3 FORWARD() {
		return Vector3(0, 0, 1);
	}

	__host__ __device__ inline static float dot(const Vector3& v1, const Vector3& v2) {
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}

	__host__ __device__ inline static Vector3 cross(const Vector3& v1, const Vector3& v2) {
		return Vector3((v1.y * v2.z - v1.z * v2.y), - (v1.x * v2.z - v1.z * v2.x), (v1.x * v2.y - v1.y * v2.x));
	}



	__host__ __device__ inline void normalize() {

		float l = length();

		if (l == 0) return;

		x /= l;
		y /= l;
		z /= l;
	}

	__host__ __device__ inline Vector3 normalized() {
		if (length() == 0) return Vector3(x, y, z);

		return Vector3(x, y, z) / length();
	}

	__host__ __device__ inline Vector3 operator*(const Vector3& v1) {
		return Vector3(v1.x * x, v1.y * y, v1.z * z);
	}

	__host__ __device__ inline Vector3 operator/(const Vector3& v1) {
		return Vector3(x / v1.x, y / v1.y, z / v1.z);
	}


	__host__ __device__ inline static Vector3 lerp(const Vector3& v1, const Vector3& v2, float amount);
	__host__ __device__ inline static Vector3 lerp2D(const Vector3& v1, const Vector3& v2, const Vector3& v3, const Vector3& v4, float amountX, float amountY);

	__host__ __device__ inline void print() {
		printf("x: %.3f, y: %.3f, z: %.3f.", x, y, z);
	}

};

__host__ __device__ inline Vector3 operator-(const Vector3& v1, const Vector3& v2) {
	return Vector3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__host__ __device__ inline Vector3 operator+(const Vector3& v1, const Vector3& v2) {
	return Vector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ inline Vector3 operator+(const Vector3& v1, const float s) {
	return Vector3(v1.x + s, v1.y + s, v1.z + s);
}

__host__ __device__ inline Vector3 operator*(const Vector3& v1, const float s) {
	return Vector3(v1.x * s, v1.y * s, v1.z * s);
}

__host__ __device__ inline Vector3 operator*(const float s, const Vector3& v) {
	return Vector3(v.x * s, v.y * s, v.z * s);
}
__host__ __device__ inline Vector3 operator/(const Vector3& v1, const float s) {
	return Vector3(v1.x / s, v1.y / s, v1.z / s);
}

__host__ __device__ inline Vector3 reflect(Vector3 v1, Vector3 v2) {
	return v1 - 2 * (Vector3::dot(v1, v2)) * v2;
}


__host__ __device__ inline bool operator==(const Vector3& v1, const Vector3& v2) {

	float EPSILON = 0.0001;

	return abs(v1.x - v2.x) < EPSILON && abs(v1.y - v2.y) < EPSILON && abs(v1.z - v2.z) < EPSILON;
}


__host__ __device__ inline Vector3 Vector3::lerp(const Vector3& v1, const Vector3& v2, float amount) {
	return v1 * amount + (1 - amount) * v2;
}

__host__ __device__ inline Vector3 Vector3::lerp2D(const Vector3& v1, const Vector3& v2, const Vector3& v3, const Vector3& v4, float amountX, float amountY) {
	return lerp(lerp(v1, v2, amountX), lerp(v3, v4, amountX), amountY);
}



#endif