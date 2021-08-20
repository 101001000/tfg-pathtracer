#ifndef TEXTURE_H
#define TEXTURE_H

#include "Math.h"

enum Filter { NO_FILTER, BILINEAR };

class Texture {


/*
    TODO:
        sustituir la inicialización por memset
        comprobar que pasa con texturas de distinto tamaño, redimensionar(?)
*/

public:

	float* data;
    bool USE_IMAGE;

    Filter filter = BILINEAR;

    Vector3 color;

    unsigned int width;
    unsigned int height;

    float xTile = 1;
    float yTile = 1;

    float xOffset = 0;
    float yOffset = 0;

public:

	__host__ Texture(const char* filepath) {

        USE_IMAGE = true;

        int i;
        FILE* f = fopen(filepath, "rb");
        unsigned char info[54];

        // read the 54-byte header
        fread(info, sizeof(unsigned char), 54, f);

        // extract image height and width from header
        width = *(int*)&info[18];
        height = *(int*)&info[22];

        data = new float[width * height * 3];

        unsigned char* tmpData = new unsigned char[width * height * 3];

        for (int i = 0; i < width * height * 3; i++) data[i] = 0;

        // allocate 3 bytes per pixel
        int size = 3 * width * height;

        // read the rest of the data at once
        fread(tmpData, sizeof(unsigned char), size, f);
        fclose(f);

        for (i = 0; i < size; i += 3) {
            // flip the order of every 3 bytes
            float tmp = tmpData[i];
            tmpData[i] = tmpData[i + 2];
            tmpData[i + 2] = tmp;
        }

        for (i = 0; i < size; i ++) {
            data[i] = ((float)tmpData[i]) / 256.0;
        }
	}

    __host__ __device__ Texture(Vector3 _color) {

        width = 1;
        height = 1;

        USE_IMAGE = false;
        color = _color;
        data = new float[3];
        data[0] = color.x; data[1] = color.y; data[2] = color.z;
    }

    __host__ __device__ Texture() {
        
        width = 1;
        height = 1;

        USE_IMAGE = false;

        data = new float[3];

        data[0] = 0; data[1] = 0; data[2] = 0;
    }

    __host__ __device__ Vector3 getValueFromCoordinates(int x, int y) {

        Vector3 pixel;

        // Offset and tiling tranformations
        x = (int)(xTile * (x + xOffset)) % width;
        y = (int)(yTile * (y + yOffset)) % height;

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

    __host__ __device__ static inline Vector3 reverseSphericalMapping(float u, float v) {

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