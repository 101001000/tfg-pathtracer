#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Sphere.h"
#include "Camera.h"
#include "Scene.h"

#pragma once

int getBuffer(float *pixelBuffer, int *pathcountBuffer, int size);

struct HitData {

    float metallic;
    float roughness;

	float clearcoatGloss;
	float clearcoat;
	float anisotropic;
	float eta;
	float transmission;
	float specular;
	float specularTint;
	float sheenTint;
	float subsurface;
	float sheen;

    Vector3 emission;
    Vector3 albedo;


    Vector3 normal;
    Vector3 tangent;
    Vector3 bitangent;
};

void printPdfMaterial(Material material, int samples);
void printBRDFMaterial(Material material, int samples);
void printHDRISampling(HDRI hdri, int samples);

void renderCuda(Scene* scene, int sampleTarget);

int renderSetup(Scene *scene);

int getSamples();

