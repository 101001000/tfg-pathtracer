#pragma once

#include "Scene.hpp"
#include "RSJparser.hpp"

// TODO some cleaning
static Scene loadScene(std::string path) {

	ObjLoader objLoader;

	path += "\\";

	printf("Loading scene \n");

	Scene scene = Scene();

	std::ifstream st(path + "scene.json");
	std::string str((std::istreambuf_iterator<char>(st)),
		std::istreambuf_iterator<char>());


	RSJresource scene_json(str);

	// Camera
	RSJresource camera_json = scene_json["camera"].as<RSJresource>();
	RSJresource camera_pos_json = camera_json["position"].as<RSJresource>();
	RSJresource camera_rot_json = camera_json["rotation"].as<RSJresource>();

	int xRes = camera_json["xRes"].as<int>();
	int yRes = camera_json["yRes"].as<int>();

	float focalLength = camera_json["focalLength"].as<double>();
	float focusDistance = camera_json["focusDistance"].as<double>();
	float aperture = camera_json["aperture"].as<double>();
	float bokeh = camera_json["bokeh"].as<bool>();

	Vector3 cameraPosition = Vector3(camera_pos_json["x"].as<double>(), camera_pos_json["y"].as<double>(), camera_pos_json["z"].as<double>());
	Vector3 cameraRotation = Vector3(camera_rot_json["x"].as<double>(), camera_rot_json["y"].as<double>(), camera_rot_json["z"].as<double>());

	scene.camera = Camera(xRes, yRes);
	scene.camera.focalLength = focalLength;
	scene.camera.focusDistance = focusDistance;
	scene.camera.aperture = aperture;
	scene.camera.position = cameraPosition;
	scene.camera.rotation = cameraRotation;
	scene.camera.bokeh = bokeh;


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

	//Materials v2
	printf("Loading materials\n");
	std::vector<UnloadedMaterial> umtls = objLoader.loadMtls(path + "scene.mtl");

	for (int i = 0; i < umtls.size(); i++) {

		for (const auto& map : umtls[i].maps) {
			std::string key = map.first;
			std::string mapPath = umtls[i].maps[key];

			CS colorSpace = CS::LINEAR;

			int textureId = scene.textureCount();
			bool dupTex = false;

			for (int j = 0; j < scene.textures.size(); j++) {
				if (scene.textures[j].path == mapPath) {
					textureId = j;
					dupTex = true;
				}
			}

			if (key == "map_Kd") {
				umtls[i].mat.albedoTextureID = textureId;
				colorSpace = CS::sRGB;
			}

			if (key == "map_Ns")
				umtls[i].mat.roughnessTextureID = textureId;

			if (key == "refl")
				umtls[i].mat.metallicTextureID = textureId;

			if (key == "map_Bump")
				umtls[i].mat.normalTextureID = textureId;

			if(!dupTex)
				scene.addTexture(Texture(mapPath, colorSpace));
		}
		scene.addMaterial(umtls[i].mat);
	}

	//Objects v2
	printf("Loading objects\n");

	std::vector<MeshObject> objects = objLoader.loadObjs(path + "scene.obj");

	for (int i = 0; i < objects.size(); i++) {

		objects[i].materialID = 0;

		for (int j = 0; j < scene.materials.size(); j++) {
			if (scene.materials[j].name == objects[i].matName)
				objects[i].materialID = j;
		}
		scene.addMeshObject(objects[i]);
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