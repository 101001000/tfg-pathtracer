#ifndef MATH_H
#define MATH_H

#include "Vector.h"
#include "cuda_runtime.h"

#define PI 3.14159265358979323846f

#define FAST_LERP

__host__ __device__ static float map(float a, float b, float c, float d, float e) {
    return d + ((a - b) / (c - b)) * (e - d);
}

__host__ __device__ static float clamp(float a, float b, float c) {
    return a < b ? b : a > c ? c : a;
}

__host__ __device__ static Vector3 clamp(Vector3 v, float b, float c) {
    return Vector3(clamp(v.x, b, c), clamp(v.y, b, c), clamp(v.z, b, c));
}

__host__ __device__ static float limitUV(float u) {
    u -= (int) u;
    if (u < 0) u = 1 + u;
    return u;
}

__host__ __device__ static float lerp(float a, float b, float c) {
#ifdef FAST_LERP
    // Este método no se comporta bien en algunos casos de infinitos y nans
    return a + c * (b - a);
#else
    return b * c + a * (1 - c);
#endif
}

__host__ __device__ static Vector3 sqrt(Vector3 v) {
    return Vector3(sqrt(v.x), sqrt(v.y), sqrt(v.z));
}

__host__ __device__ static Vector3 lerp(Vector3 a, Vector3 b, float c) {
    return Vector3(lerp(a.x, b.x, c), lerp(a.y, b.y, c), lerp(a.z, b.z, c));
}

__host__ __device__ static float minf(float a, float b) {
    return a < b ? a : b;
}

__host__ __device__ static float maxf(float a, float b) {
    return a > b ? a : b;
}



#endif