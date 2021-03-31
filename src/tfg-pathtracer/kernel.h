#include "cuda_runtime.h"
#include "Sphere.h"
#include "Camera.h"
#include "Scene.h"
#include "device_launch_parameters.h"

#pragma once

cudaError_t getBuffer(float* bufffer, int size);

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


void renderCuda(Scene* scene);

cudaError_t renderSetup(Scene* scene);

int getSamples();

