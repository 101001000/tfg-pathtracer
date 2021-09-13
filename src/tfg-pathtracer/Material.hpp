#include "Vector.hpp"
#include "Texture.hpp"

#ifndef MATERIAL_H
#define MATERIAL_H


class Material {

public:

	int albedoTextureID = -1;
	int emissionTextureID = -1;
	int roughnessTextureID = -1;
	int metallicTextureID = -1;
	int normalTextureID = -1;

	Vector3 albedo = Vector3(0.5,0.5,0.5);
	Vector3 emission = Vector3::Zero();;

	float roughness = 1;
	float metallic = 0;
	float clearcoatGloss = 0;
	float clearcoat = 0;
	float anisotropic = 0;
	float eta = 0;
	float transmission = 0;
	float specular = 0.5;
	float specularTint = 0;
	float sheenTint = 0;
	float subsurface = 0;
	float sheen = 0;

};



#endif