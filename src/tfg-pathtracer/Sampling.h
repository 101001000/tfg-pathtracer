#include "Vector.h"
#include "Math.h"

//TODO naming convention

__device__ __host__ inline Vector3 uniformSampleSphere(float u1, float u2) {

    float z = 1.0 - 2.0 * u1;
    float r = sqrt(maxf(0.f, 1.0 - z * z));
    float phi = 2.0 * PI * u2;
    float x = r * cos(phi);
    float y = r * sin(phi);

    return Vector3(x, y, z);
}

__device__ __host__ inline void uniformCircleSampling(float u1, float u2, float u3, float& x, float& y) {

    float t = 2 * PI * u1;
    float u = u2 + u3;
    float r = u > 1 ? 2 - u : u;
       
    x = r * cos(t);
    y = r * sin(t);
}

__device__ __host__ inline Vector3 CosineSampleHemisphere(float u1, float u2){

    Vector3 dir;
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    dir.x = r * cos(phi);
    dir.y = r * sin(phi);
    dir.z = sqrt(maxf(0.0, 1.0 - dir.x * dir.x - dir.y * dir.y));

    return dir;
}

__device__ __host__ inline Vector3 ImportanceSampleGGX(float rgh, float r1, float r2) {
    float a = maxf(0.001, rgh);

    float phi = r1 * PI * 2;

    float cosTheta = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
    float sinTheta = clamp(sqrt(1.0 - (cosTheta * cosTheta)), 0.0, 1.0);
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);

    return Vector3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}