#ifndef MATH_H
#define MATH_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Vector.h"

#define PI 3.14159265358979323846f

#define FAST_LERP

static float map(float a, float b, float c, float d, float e) {
    return d + ((a - b) / (c - b)) * (e - d);
}

static float clamp(float a, float b, float c) {
    return a < b ? b : a > c ? c : a;
}

static Vector3 clamp(Vector3 v, float b, float c) {
    return Vector3(clamp(v.x, b, c), clamp(v.y, b, c), clamp(v.z, b, c));
}

static float limitUV(float u) {
    u -= (int) u;
    if (u < 0) u = 1 + u;
    return u;
}

static float lerp(float a, float b, float c) {
#ifdef FAST_LERP
    return a + c * (b - a);
#else
    return b * c + a * (1 - c);
#endif
}

static Vector3 sqrt(Vector3 v) {
    return Vector3(sycl::sqrt(v.x), sycl::sqrt(v.y), sycl::sqrt(v.z));
}

static Vector3 lerp(Vector3 a, Vector3 b, float c) {
    return Vector3(lerp(a.x, b.x, c), lerp(a.y, b.y, c), lerp(a.z, b.z, c));
}

static float minf(float a, float b) {
    return a < b ? a : b;
}

static float maxf(float a, float b) {
    return a > b ? a : b;
}



#endif