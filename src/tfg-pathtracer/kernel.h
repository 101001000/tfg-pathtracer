#include "Camera.hpp"
#include "Scene.hpp"
#include "HDRI.hpp"

#pragma once

enum Passes {BEAUTY, DENOISE, NORMAL, TANGENT, BITANGENT};

struct RenderParameters {

	unsigned int width, height;
	unsigned int sampleTarget;

	bool passes[PASSES_COUNT];

	RenderParameters(unsigned int width, unsigned int height, unsigned int sampleTarget) : width(width), height(height), sampleTarget(sampleTarget) {
		passes[BEAUTY] = true;
		passes[DENOISE] = true;
		passes[NORMAL] = true;
		passes[TANGENT] = true;
		passes[BITANGENT] = true;
	};
	RenderParameters() : width(1280), height(720), sampleTarget(100) {};
};
struct RenderData {

	RenderParameters pars;

	float* passes[PASSES_COUNT];

	size_t freeMemory = 0;
	size_t totalMemory = 0;

	int pathCount = 0;
	int samples = 0;

	std::chrono::steady_clock::time_point startTime;

	RenderData() {};

	~RenderData() {

	};
};

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
void calcNormalPass();

cudaError_t renderCuda(Scene* scene, int sampleTarget);

cudaError_t renderSetup(Scene* scene);

cudaError_t getBuffers(RenderData& renderData, int* pathcountBuffer, int size);

int getSamples();

