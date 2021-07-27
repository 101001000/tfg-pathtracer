#ifndef SCENE_H
#define SCENE_H

#include "Material.h"
#include "MeshObject.h"
#include "Tri.h"
#include "Camera.h"
#include "Sphere.h"
#include "BVH.h"
#include "PointLight.h"
#include "HdrLoader.h"
#include <vector>
#include "HDRI.h"

class Scene {

public:

	std::vector<Material> materials;
	std::vector<Sphere> spheres;
	std::vector<MeshObject> meshObjects;
	std::vector<Texture> textures;
	std::vector<Tri> tris;
	std::vector<PointLight> pointLights;

	HDRI hdri;
	Camera camera;

public:

	Scene() {

	}

	int materialCount() {
		return materials.size();
	}

	int sphereCount() {
		return spheres.size();
	}

	int textureCount() {
		return textures.size();
	}

	int meshObjectCount() {
		return meshObjects.size();
	}

	int triCount() {
		return tris.size();
	}

	int pointLightCount() {
		return pointLights.size();
	}

	Camera* getMainCamera() {
		return &camera;
	}

	Material* getMaterials() {
		if (materials.size() == 0) return (Material*)0;
		return materials.data();
	}

	Sphere* getSpheres() {
		if(spheres.size() == 0) return (Sphere*)0;
		return spheres.data();
	}

	Tri* getTris() {
		if (tris.size() == 0) return (Tri*)0;
		return tris.data();
	}

	MeshObject* getMeshObjects() {
		if (meshObjects.size() == 0) return (MeshObject*)0;
		return meshObjects.data();
	}

	PointLight* getPointLights() {
		if (pointLights.size() == 0) return (PointLight*)0;
		return pointLights.data();
	}

	Texture* getTextures() {
		if (textures.size() == 0) return (Texture*)0;
		return textures.data();
	}

	void addSphere(Sphere sphere) {
		sphere.objectID = sphereCount();
		spheres.push_back(sphere);
	}

	void addPointLight(PointLight pointLight) {
		pointLights.push_back(pointLight);
	}

	void addTexture(Texture texture) {
		textures.push_back(texture);
	}


	void addMaterial(Material material) {
		materials.push_back(material);
	}

	void addMeshObject(MeshObject meshObject) {

		meshObject.objectID = meshObjectCount();

		for (int i = 0; i < meshObject.triCount; i++) {
			meshObject.tris[i].objectID = meshObject.objectID;
			tris.push_back(meshObject.tris[i]);
		}

		meshObjects.push_back(meshObject);

		//Update ptrs
		int sum = 0;

		for (int i = 0; i < meshObjects.size(); i++) {
			meshObjects.at(i).tris = tris.data() + sum;
			sum += meshObjects.at(i).triCount;
		}
	}



	void addHDRI(const char* filepath) {

		hdri = HDRI(filepath);
	}

	void addHDRI(Vector3 color) {

		hdri = HDRI(color);
	}

	BVH* buildBVH() {

		printf("\nBuilding BVH with DEPTH=%d and SAHBINS=%d \n", DEPTH, SAHBINS);

		auto t1 = std::chrono::high_resolution_clock::now();

		BVH* bvh = new BVH();

		int* triIndices = new int[triCount()];

		bvh->triIndices = triIndices;

		bvh->build(&tris);

		auto t2 = std::chrono::high_resolution_clock::now();

		auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

		printf("\nBVH built in %dms\n\n", ms_int);

		return bvh;
	}



};


#endif