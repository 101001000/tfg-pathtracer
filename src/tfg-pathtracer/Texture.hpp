#ifndef TEXTURE_H
#define TEXTURE_H

#if !defined(__CUDACC__)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif


#include "Math.hpp"

enum Filter { NO_FILTER, BILINEAR };
enum CS { LINEAR, sRGB };

class Texture {

public:

	float* data;

    Filter filter = NO_FILTER;
    Vector3 color;

    std::string path;

    int width;
    int height;

    float xTile = 1;
    float yTile = 1;

    float xOffset = 0;
    float yOffset = 0;

public:

    __host__ Texture(std::string filepath) : Texture(filepath, CS::sRGB) {}

    //TODO fix this compilation nightmare
#if !defined(__CUDACC__)
    __host__ Texture(std::string filepath, CS colorSpace) {

        stbi_ldr_to_hdr_gamma(1.0f);
        stbi_set_flip_vertically_on_load(true);

        printf("Loading texture from %s... ", filepath.c_str());

        //TODO colorspace fix
        path = filepath;

        int channels;
        float* tmp_data = stbi_loadf(filepath.c_str(), &width, &height, &channels, 0);

        data = new float[width * height * 3];

        for (int i = 0; i < width * height * 3; i++) {
            data[i] = ((float)tmp_data[i]);

            if (colorSpace == CS::sRGB)
                data[i] = fastPow(data[i], 2.2);
        }

        printf("Loaded! %dpx x %dpx, %d channels\n", width, height, channels);
        stbi_image_free(tmp_data);
    }
#else
    __host__ Texture(std::string filepath, CS colorSpace) {
        printf("COMPILATION MISMATCH");
    }
       
#endif
 
    __host__ __device__ Texture(Vector3 _color) {

        width = 1; height = 1;

        color = _color;
        data = new float[3];
        data[0] = color.x; data[1] = color.y; data[2] = color.z;
    }

    __host__ __device__ Texture() {
        
        width = 1;
        height = 1;

        data = new float[3];

        data[0] = 0; data[1] = 0; data[2] = 0;
    }

    __host__ __device__ Vector3 getValueFromCoordinates(int x, int y) {

        Vector3 pixel;

        // Offset and tiling tranformations
        x = (int)(xTile * (x + xOffset * width)) % width;
        y = (int)(yTile * (y + yOffset * height)) % height;

        pixel.x = data[(3 * (y * width + x) + 0)];
        pixel.y = data[(3 * (y * width + x) + 1)];
        pixel.z = data[(3 * (y * width + x) + 2)];

        return pixel;
    }

    __host__ __device__ Vector3 getValueFromUV(float u, float v) {
        return getValueFromCoordinates(u * width, v * height);
    }

    __host__ __device__ Vector3 getValueBilinear(float u, float v) {
        
        float x = u * width;
        float y = v * height;

        float t1x = floor(x);
        float t1y = floor(y);

        float t2x = t1x + 1;
        float t2y = t1y + 1;

        float a = (x - t1x) / (t2x - t1x);
        float b = (y - t1y) / (t2y - t1y);

        Vector3 v1 = getValueFromCoordinates(t1x, t1y);
        Vector3 v2 = getValueFromCoordinates(t2x, t1y);
        Vector3 v3 = getValueFromCoordinates(t1x, t2y);
        Vector3 v4 = getValueFromCoordinates(t2x, t2y);

        // Linear interpolation
        return lerp(lerp(v1, v2, a), lerp(v3, v4, a), b);
	}

    __host__ __device__ Vector3 getValueFromUVFiltered(float u, float v) {
        if (filter == NO_FILTER)
            return getValueFromUV(u, v);
        else if (filter == BILINEAR)
            return getValueBilinear(u, v);
    }

    __host__ __device__ static inline void sphericalMapping(Vector3 origin, Vector3 point, float radius, float& u, float& v) {

        // Point is normalized to radius 1 sphere
        Vector3 p = (point - origin) / radius;

        float theta = acos(-p.y);
        float phi = atan2(-p.z, p.x) + PI;

        u = phi / (2 * PI);
        v = theta / PI;

        limitUV(u,v);
    }

    __host__ __device__ inline Vector3 transformUV(float u, float v) {

        int x = u * width;
        int y = v * height;

        // OJO TILE

        x = (int)(xTile * (x + xOffset * width)) % width;
        y = (int)(yTile * (y + yOffset * height)) % height;


        float nu = (float)x / (float)width;
        float nv = (float)y / (float)height;

        limitUV(nu, nv);

        return Vector3(nu, nv, 0);
    }

    __host__ __device__ inline Vector3 inverseTransformUV(float u, float v) {

        int x = u * width;
        int y = v * height;

        // OJO TILE

        x = (int)(xTile * (x - xOffset * width)) % width;
        y = (int)(yTile * (y - yOffset * height)) % height;

        float nu = (float)x / (float)width;
        float nv = (float)y / (float)height;

        limitUV(nu, nv);

        return Vector3(nu, nv, 0);
    }

    __host__ __device__ inline Vector3 reverseSphericalMapping(float u, float v) {

        float phi = u * 2 * PI;
        float theta = v * PI;

        float px = cos(phi - PI);
        float py = -cos(theta);
        float pz = -sin(phi - PI);

        float a = sqrt(1 - py * py);

        return Vector3(a * px, py, a * pz);
    }
};


#endif