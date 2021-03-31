#ifndef MATH_H
#define MATH_H

#include "Vector.h"
#include "cuda_runtime.h"

#define PI 3.14159265358979323846f

#define FAST_LERP

__host__ __device__ static float clamp(float a, float b, float c) {
    if (a < b) return b;
    if (a > c) return c;
    return a;
}

__host__ __device__ static Vector3 clamp(Vector3 v, float b, float c) {
    
    if (v.x < b) v.x = b;
    if (v.y < b) v.y = b;
    if (v.z < b) v.z = b;

    if (v.x > c) v.x = c;
    if (v.y > c) v.y = c;
    if (v.z > c) v.z = c;

    return v;
}

//TODO: Buscar el nombre apropiado para esta función
__host__ __device__ static float circle(float v) {
    v -= (int) v;
    if (v < 0) v = 1 + v;

    return v;
}


__host__ __device__ static float lerp(float a, float b, float c) {

#ifdef FAST_LERP
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
    if (a < b) return a;
    return b;
}

__host__ __device__ static float maxf(float a, float b) {
    if (a > b) return a;
    return b;
}



#endif