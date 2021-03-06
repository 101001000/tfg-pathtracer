#ifndef TEXTURE_H
#define TEXTURE_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Math.h"
#include <math.h>

class Texture {


/*
    TODO:
        sustituir la inicialización por memset
        comprobar que pasa con texturas de distinto tamaño, redimensionar(?)
*/

public:

	float* data;
    bool USE_IMAGE;

    Vector3 color;

    float multiply = 1;

    unsigned int width;
    unsigned int height;

    float xTile = 1;
    float yTile = 1;

    float xOffset = 0;
    float yOffset = 0;

public:

	Texture(const char* filepath) {

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

    Texture(Vector3 _color) {

        width = 1;
        height = 1;

        USE_IMAGE = false;

        color = _color;

        data = new float[3];

        data[0] = color.x;
        data[1] = color.y;
        data[2] = color.z;
    }

    Texture() {
        
        width = 1;
        height = 1;

        USE_IMAGE = false;

        data = new float[3];

        data[0] = 0;
        data[1] = 0;
        data[2] = 0;

    }

    Vector3 getValue(int x, int y) {

        Vector3 pixel;

        x *= xTile;
        y *= yTile;

        x %= width;
        y %= height;

        pixel.x = data[(3 * (y * width + x) + 0)];
        pixel.y = data[(3 * (y * width + x) + 1)];
        pixel.z = data[(3 * (y * width + x) + 2)];

        return pixel * multiply;
    }

    Vector3 getValueBilinear(float u, float v) {
        
        u = ((u + xOffset) * xTile);
        v = ((v + yOffset) * yTile);

        u = limitUV(u);
        v = limitUV(v);

        float x = u * width;
        float y = v * height;

        float t1x = sycl::floor(x);
        float t1y = sycl::floor(y);

        float t2x = t1x + 1;
        float t2y = t1y + 1;

        float a = (x - t1x) / (t2x - t1x);
        float b = (y - t1y) / (t2y - t1y);

        Vector3 v1 = getValue(t1x, t1y);
        Vector3 v2 = getValue(t2x, t1y);
        Vector3 v3 = getValue(t1x, t2y);
        Vector3 v4 = getValue(t2x, t2y);

        return lerp(lerp(v1, v2, a), lerp(v3, v4, a), b);
	}

    // Calcula las coordenadas UV para un punto en una esfera con cierto radio

    static inline Vector3 reverseSphericalMapping(float u, float v) {

        float phi = u * 2 * PI;
        float theta = v * PI;

        float py = -sycl::cos(theta);

        float pz = -sycl::sin(phi - PI);
        float px = sycl::cos(phi - PI);

        float a = sycl::sqrt(1 - py * py);

        return Vector3(a*px, py, a*pz);     
    }

    static inline void sphericalMapping(Vector3 origin, Vector3 point, float radius, float& u, float& v) {

        Vector3 p = (point - origin) / radius;

        float theta = sycl::acos(-p.y);
        float phi = sycl::atan2(-p.z, p.x) + PI;

        u = phi / (2 * PI);
        v = theta / PI;

        u += -(u > 1) + -(u < 0);
        v += -(v > 1) + -(v < 0);

        /*

        if (u > 1) u -= 1;
        if (v > 1) v -= 1;

        if (u < 0) u += 1;
        if (v < 0) v += 1;

        */
    }
};


#endif