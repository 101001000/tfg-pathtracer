#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "Camera.h"
#include "Scene.h"
#include "test.h" intonce

int geintat* pixelBuffer, int* pathcountBuffer, int size);

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

void renderCuda(Scene *scene, intintet);

int reintcene* scene);

int getSamples();

