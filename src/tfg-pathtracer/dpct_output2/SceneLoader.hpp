#pragma once

#include "Scene.hpp"
#include "RSJparser.hpp"

static Scene loadScene(std::string path) {

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

				//Maybe I need to adjust colorspace here

				switch (j) {
				case 0:
					material.albedo = json_data.as<Vector3>();
				case 1:
					material.emission = json_data.as<Vector3>();
				case 2:
					material.roughness = json_data.as<double>();
				case 3:
					material.metallic = json_data.as<double>();
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