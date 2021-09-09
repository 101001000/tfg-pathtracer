#ifndef SCENE_H
#define SCENE_H

#include <thread>         
#include <chrono>    
#include <iostream>
#include <Windows.h>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <map>

#include "Material.hpp"
#include "Tri.hpp"
#include "Camera.hpp"
#include "BVH.hpp"
#include "PointLight.hpp"
#include "HdrLoader.hpp"
#include "HDRI.hpp"
#include "RSJparser.hpp"
#include "ObjLoader.hpp"

class Scene {

public:

	std::vector<Material> materials;
	std::vector<MeshObject> meshObjects;
	std::vector<Texture> textures;
	std::vector<Tri> tris;
	std::vector<PointLight> pointLights;

	HDRI hdri;
	Camera camera;

public:

	Scene() {}

	static Scene sceneBuilder(std::string path) {

		printf("Loading scene \n");

		Scene scene = Scene();

		std::ifstream st(path + "scene.json");
		std::string str((std::istreambuf_iterator<char>(st)),
			std::istreambuf_iterator<char>());


		RSJresource scene_json(str);


		// Camera
		RSJresource camera_json = scene_json["camera"].as<RSJresource>();
		RSJresource camera_pos_json = camera_json["position"].as<RSJresource>();

		int xRes = camera_json["xRes"].as<int>();
		int yRes = camera_json["yRes"].as<int>();

		float focalLength = camera_json["focalLength"].as<double>();
		float focusDistance = camera_json["focusDistance"].as<double>();
		float aperture = camera_json["aperture"].as<double>();

		Vector3 cameraPosition = Vector3(camera_pos_json["x"].as<double>(), camera_pos_json["y"].as<double>(), camera_pos_json["z"].as<double>());

		scene.camera = Camera(xRes, yRes);
		scene.camera.focalLength = focalLength;
		scene.camera.focusDistance = focusDistance;
		scene.camera.aperture = aperture;
		scene.camera.position = cameraPosition;


		// HDRI

		RSJresource hdri_json = scene_json["hdri"].as<RSJresource>();

		if (hdri_json["name"].exists()) {
			scene.addHDRI(path + "HDRI\\" + hdri_json["name"].as<std::string>() + ".hdr");
		}
		else if (hdri_json["color"].exists()) {
			Vector3 color = Vector3(hdri_json["color"]["r"].as<double>(), hdri_json["color"]["g"].as<double>(), hdri_json["color"]["b"].as<double>());
			scene.addHDRI(color);
		}

		if (hdri_json["xOffset"].exists())
			scene.hdri.texture.xOffset = hdri_json["xOffset"].as<double>();

		if (hdri_json["yOffset"].exists())
			scene.hdri.texture.xOffset = hdri_json["yOffset"].as<double>();


		// Materials

		std::map<std::string, int> matIds;

		RSJarray materials_json = scene_json["materials"].as<RSJarray>();

		for (int i = 0; i < materials_json.size(); i++) {

			std::string name = materials_json[i]["name"].as<std::string>();

			printf("Loading material %s\n", name.c_str());

			Material material = Material();

			std::vector<std::string> mapnames = { "albedo", "emission", "roughness", "metallic", "normal" };

			//@todo add support for the rest of parameters

			for (int j = 0; j < mapnames.size(); j++) {

				std::ifstream f(path + "Textures\\" + name + "_" + mapnames[j] + ".bmp");

				if (f.good()) {

					CS colorSpace = CS::LINEAR;

					switch (j) {
					case 0:
						material.albedoTextureID = scene.textureCount();
						colorSpace = CS::sRGB;
					case 1:
						material.emissionTextureID = scene.textureCount();
						colorSpace = CS::sRGB;
					case 2:
						material.roughnessTextureID = scene.textureCount();
					case 3:
						material.metallicTextureID = scene.textureCount();
					case 4:
						material.normalTextureID = scene.textureCount();
					}

					scene.addTexture(Texture(path + "Textures\\" + name + "_" + mapnames[j] + ".bmp", colorSpace));
				}
				else if (materials_json[i][mapnames[j]].exists()) {
					RSJresource json_data = materials_json[i][mapnames[j]];

					Vector3 color = Vector3(json_data["r"].as<double>(), json_data["g"].as<double>(), json_data["b"].as<double>());

					//Maybe I need to adjust colorspace here

					switch (j) {
					case 0:
						material.albedo = color;
					case 1:
						material.emission = color;
					case 2:
						material.roughness = color.x;
					case 3:
						material.metallic = color.x;
					}
				}
			}

			matIds[name] = scene.materialCount();
			scene.addMaterial(material);
		}


		// Objects

		RSJarray objects_json = scene_json["objects"].as<RSJarray>();

		for (int i = 0; i < objects_json.size(); i++) {

			std::string name = objects_json[i]["name"].as<std::string>();
			std::string matName = objects_json[i]["material"].as<std::string>();

			printf("Loading object %s\n", name.c_str());

			MeshObject object = ObjLoader::loadObj(path + "Objects\\" + name + ".obj");

			object.materialID = matIds[matName];

			scene.addMeshObject(object);
		}

		// PointLights

		RSJarray pointLight_json = scene_json["pointLights"].as<RSJarray>();

		for (int i = 0; i < pointLight_json.size(); i++) {

			Vector3 position = pointLight_json[i]["position"].as<Vector3>();
			Vector3 radiance = pointLight_json[i]["radiance"].as<Vector3>();

			scene.addPointLight(PointLight(position, radiance));

			printf("Loaded pointLight pos ");
			position.print();
			printf(" radiance ");
			radiance.print();
			printf("\n");
		}

		return scene;
	}

	int materialCount() { return materials.size(); }
	int textureCount() { return textures.size(); }
	int meshObjectCount() {	return meshObjects.size(); }
	int triCount() { return tris.size(); }
	int pointLightCount() {	return pointLights.size();}

	Camera* getMainCamera() { return &camera; }

	Material* getMaterials() {
		if (materials.size() == 0) return (Material*)0;
		return materials.data();
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

	void addPointLight(PointLight pointLight) {	pointLights.push_back(pointLight); }
	void addTexture(Texture texture) { textures.push_back(texture); }
	void addMaterial(Material material) { materials.push_back(material); }

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

	void addHDRI(std::string filepath) { hdri = HDRI(filepath); }
	void addHDRI(Vector3 color) { hdri = HDRI(color); }

	BVH* buildBVH() {

		printf("\nBuilding BVH with DEPTH=%d and SAHBINS=%d \n", BVH_DEPTH, BVH_SAHBINS);

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