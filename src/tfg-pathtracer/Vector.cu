#include "Vector.h"

inline __host__ __device__ Vector3 Vector3::lerp(const Vector3& v1, const Vector3& v2, float amount)
{
    return amount * v1 + (1 - amount) * v2;
}
