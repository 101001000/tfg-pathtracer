#ifndef TEXTURE_H
#define TEXTURE_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Math.hpp"

enum Filter { NO_FILTER, BILINEAR };
enum CS { LINEAR, sRGB };

class Texture {


/*
    TODO:
        comprobar que pasa con texturas de distinto tama�o, redimensionar(?)
*/

public:

	float* data;

    Filter filter = NO_FILTER;
    Vector3 color;

    unsigned int width;
    unsigned int height;

    float xTile = 1;
    float yTile = 1;

    float xOffset = 0;
    float yOffset = 0;

public:

    Texture(std::string filepath) : Texture(filepath, CS::sRGB) {}

    	Texture(std::string filepath, CS colorSpace) {

        printf("Loading texture from file %s\n", filepath.c_str());

        int i;
        FILE* f = fopen(filepath.c_str(), "rb");
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

            if(colorSpace == CS::sRGB)
                data[i] = fastPow(data[i], 2.2);
        }

        delete(tmpData);
	}

    Texture(Vector3 _color) {

        width = 1;
        height = 1;

        color = _color;
        data = new float[3];
        data[0] = color.x; data[1] = color.y; data[2] = color.z;
    }

    Texture() {
        
        width = 1;
        height = 1;

        data = new float[3];

        data[0] = 0; data[1] = 0; data[2] = 0;
    }

    Vector3 getValueFromCoordinates(int x, int y) {

        Vector3 pixel;

        // Offset and tiling tranformations
        x = (int)(xTile * (x + xOffset * width)) % width;
        y = (int)(yTile * (y + yOffset * height)) % height;

        pixel.x = data[(3 * (y * width + x) + 0)];
        pixel.y = data[(3 * (y * width + x) + 1)];
        pixel.z = data[(3 * (y * width + x) + 2)];

        return pixel;
    }

    Vector3 getValueFromUV(float u, float v) {
        return getValueFromCoordinates(u * width, v * height);
    }

    Vector3 getValueBilinear(float u, float v) {
        
        float x = u * width;
        float y = v * height;

        float t1x = sycl::floor(x);
        float t1y = sycl::floor(y);

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

    Vector3 getValueFromUVFiltered(float u, float v) {
        if (filter == NO_FILTER)
            return getValueFromUV(u, v);
        else if (filter == BILINEAR)
            return getValueBilinear(u, v);
    }

    static inline void sphericalMapping(Vector3 origin, Vector3 point, float radius, float& u, float& v) {

        // Point is normalized to radius 1 sphere
        Vector3 p = (point - origin) / radius;

        float theta = sycl::acos(-p.y);
        float phi = sycl::atan2(-p.z, p.x) + PI;

        u = phi / (2 * PI);
        v = theta / PI;

        limitUV(u,v);
    }

    inline Vector3 transformUV(float u, float v) {

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

    inline Vector3 inverseTransformUV(float u, float v) {

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

    inline Vector3 reverseSphericalMapping(float u, float v) {

        float phi = u * 2 * PI;
        float theta = v * PI;

        float px = sycl::cos(phi - PI);
        float py = -sycl::cos(theta);
        float pz = -sycl::sin(phi - PI);

        float a = sycl::sqrt(1 - py * py);

        return Vector3(a * px, py, a * pz);
    }
};


#endif