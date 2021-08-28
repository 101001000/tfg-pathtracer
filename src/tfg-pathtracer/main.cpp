#include <SFML/Graphics.hpp>
#include <iostream>
#include <Windows.h>
#include "Camera.hpp"
#include "Ray.hpp"
#include "kernel.h"
#include "Scene.hpp"
#include <thread>         
#include <chrono>     
#include "Texture.hpp"
#include "PostProcessing.h"
#include "ObjLoader.hpp"
#include "BVH.hpp"


#define UPDATE_INTERVAL 500

std::thread t;

Scene clockScene() {

	Scene scene;

	std::string path = "C:\\Users\\Kike\\Desktop\\Scenes\\Clock\\render\\";

	scene.camera = Camera(960, 1080);
	scene.camera.position = Vector3(0, 0.08836, -0.6336);
	scene.camera.focalLength = 50 * 0.001;
	scene.camera.focusDistance = 0.59;
	scene.camera.aperture = 2.8;

	scene.addMeshObject(ObjLoader::loadObj(path + "clock.obj"));
	scene.addMaterial(Material());
	scene.addTexture(Texture(path + "clock_albedo.bmp"));
	scene.materials.at(0).albedoTextureID = 0;
	scene.addTexture(Texture(path + "clock_roughness.bmp", CS::LINEAR));
	scene.materials.at(0).roughnessTextureID = 1;
	scene.addTexture(Texture(path + "clock_normal.bmp", CS::sRGB));
	scene.materials.at(0).normalTextureID = 2;
	scene.addTexture(Texture(path + "clock_metallic.bmp", CS::LINEAR));
	scene.materials.at(0).metallicTextureID = 3;
	scene.meshObjects.at(0).materialID = 0;

	scene.addMeshObject(ObjLoader::loadObj(path + "chair.obj"));
	scene.addMaterial(Material());
	scene.addTexture(Texture(path + "chair_albedo.bmp"));
	scene.materials.at(1).albedoTextureID = 4;
	scene.addTexture(Texture(path + "chair_roughness.bmp"));
	scene.materials.at(1).roughnessTextureID = 5;
	scene.addTexture(Texture(path + "chair_normal.bmp", CS::LINEAR));
	scene.materials.at(1).normalTextureID = -1;
	scene.meshObjects.at(1).materialID = 1;


	scene.addHDRI(path + std::string("hdri.hdr"));
	scene.hdri.texture.xOffset = 0.5;

	return scene;

}

Scene cube() {


	Scene scene;

	std::string path = "C:\\Users\\Kike\\Desktop\\Scenes\\Cube\\";

	scene.camera = Camera(1280, 720);
	scene.camera.position = Vector3(0, 0, -7);
	scene.camera.focalLength = 50 * 0.001;

	scene.addMeshObject(ObjLoader::loadObj(path + "cube.obj"));
	scene.addMaterial(Material());
	scene.materials.at(0).roughness = 1;

	scene.addHDRI(Vector3(0.5));


	return scene;

}

Scene HDRIBenchmark() {

	Scene scene;

	std::string path = "C:\\Users\\Kike\\Desktop\\Scenes\\Sphere\\";

	scene.camera = Camera(1280, 720);
	scene.camera.position = Vector3(0, 0, -7);
	scene.camera.focalLength = 50 * 0.001;

	scene.addMeshObject(ObjLoader::loadObj(path + "sphere.obj"));
	scene.addMaterial(Material());
	//scene.materials.at(0).albedo = Vector3(1);
	scene.addTexture(Texture(path + "normal.bmp", CS::LINEAR));
	scene.materials.at(0).roughness = 1;
	scene.materials.at(0).metallic = 0;
	scene.meshObjects.at(0).materialID = 0;

	scene.addHDRI(path + "hdri.hdr");
	//scene.hdri.texture.xOffset = 0.5;
	//scene.addHDRI(Vector3(0.5));

	return scene;
}

Scene clementine() {

	Scene scene;

	std::string path = "C:\\Users\\Kike\\Desktop\\Scenes\\Clementine\\";

	scene.camera = Camera(1280, 720);
	scene.camera.position = Vector3(0, 0.218299, -0.988592);
	scene.camera.focalLength = 50 * 0.001;
	scene.camera.focusDistance = 0.79;
	scene.camera.aperture = 0.8;

	scene.addMeshObject(ObjLoader::loadObj(path + "clementine.obj"));
	scene.addMaterial(Material());
	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\Scenes\\Clementine\\clementine_albedo.bmp"));
	scene.materials.at(0).albedoTextureID = 0;
	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\Scenes\\Clementine\\clementine_roughness.bmp"));
	scene.materials.at(0).roughnessTextureID = 1;
	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\Scenes\\Clementine\\clementine_normal.bmp", CS::LINEAR));
	scene.materials.at(0).normalTextureID = 2;
	scene.meshObjects.at(0).materialID = 0;

	
	
	scene.addMeshObject(ObjLoader::loadObj(path + std::string("rock.obj")));
	scene.addMaterial(Material());
	scene.addTexture(Texture(path + std::string("rock_albedo.bmp")));
	scene.materials.at(1).albedoTextureID = 3;
	scene.addTexture(Texture(path + std::string("rock_roughness.bmp")));
	scene.materials.at(1).roughnessTextureID = 4;
	scene.addTexture(Texture(path + std::string("rock_normal.bmp"), CS::LINEAR));
	scene.materials.at(1).normalTextureID = 5;
	scene.meshObjects.at(1).materialID = 1;
	

	scene.addHDRI(path + std::string("hdri.hdr"));
	scene.hdri.texture.xOffset = 0.18;
	
	return scene;

}

Scene focus() {

	Scene scene;

	std::string path = "C:\\Users\\Kike\\Downloads\\scenetest\\";

	scene.camera = Camera(1280, 960);
	scene.camera.position = Vector3(0, 0, 0);
	scene.camera.focalLength = 50 * 0.001;
	scene.camera.focusDistance = 10;
	scene.camera.aperture = 0.3;

	scene.addMeshObject(ObjLoader::loadObj(path + "untitled.obj"));

	scene.addMaterial(Material());

	scene.addTexture(Texture("C:\\Users\\Kike\\Downloads\\normal.bmp", CS::LINEAR));

	scene.materials.at(0).normalTextureID = 0;

	scene.addHDRI("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\goegap_4k.hdr");

	return scene;

}

Scene cocheRefachero() {

	Scene scene;

	scene.camera = Camera(1280, 960);
	scene.camera.position = Vector3(1.739, -1.5, -8.175);
	scene.camera.focalLength = 40 * 0.001;
	scene.camera.focusDistance = 4.52;
	scene.camera.aperture = 0.6;

	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\body.obj"));
	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\chrome.obj"));
	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\rubber.obj"));
	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\silver.obj"));
	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\black.obj"));

	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\rock1.obj"));
	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\rock2.obj"));

	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\dirt1.obj"));
	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\dirt2.obj"));
	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\dirt3.obj"));
	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\road.obj"));
	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\cactus.obj"));

	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\text\\pilar_Albedo.bmp"));
	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\text\\dirt_Albedo.bmp"));
	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\text\\road_Albedo.bmp"));
	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\text\\cactus_Albedo.bmp"));

	scene.textures.at(2).yTile = 10;

	scene.addMaterial(Material());
	scene.addMaterial(Material());
	scene.addMaterial(Material());
	scene.addMaterial(Material());
	scene.addMaterial(Material());
	scene.addMaterial(Material());
	scene.addMaterial(Material());
	scene.addMaterial(Material());
	scene.addMaterial(Material());
	scene.addMaterial(Material());
	scene.addMaterial(Material());

	scene.materials.at(0).albedo = Vector3(0.3, 0.3, 0.3);
	scene.materials.at(0).metallic = 1;
	scene.materials.at(0).roughness = 0.1;

	scene.materials.at(1).albedo = Vector3(1, 1, 1);
	scene.materials.at(1).metallic = 1;
	scene.materials.at(1).roughness = 0;

	scene.materials.at(2).albedo = Vector3(0.05, 0.05, 0.05);
	scene.materials.at(2).metallic = 0;
	scene.materials.at(2).roughness = 0.3;

	scene.materials.at(3).albedo = Vector3(1, 1, 1);
	scene.materials.at(3).metallic = 1;
	scene.materials.at(3).roughness = 0.2;

	scene.materials.at(4).albedo = Vector3(0.1, 0.1, 0.1);
	scene.materials.at(4).metallic = 0;
	scene.materials.at(4).roughness = 1;

	scene.materials.at(5).albedoTextureID = 0;
	scene.materials.at(6).albedoTextureID = 1;
	scene.materials.at(7).albedoTextureID = 2;
	scene.materials.at(8).albedoTextureID = 3;

	scene.meshObjects.at(0).materialID = 0;
	scene.meshObjects.at(1).materialID = 1;
	scene.meshObjects.at(2).materialID = 2;
	scene.meshObjects.at(3).materialID = 3;
	scene.meshObjects.at(4).materialID = 4;
	scene.meshObjects.at(5).materialID = 5;
	scene.meshObjects.at(6).materialID = 5;
	scene.meshObjects.at(7).materialID = 6;
	scene.meshObjects.at(8).materialID = 6;
	scene.meshObjects.at(9).materialID = 6;
	scene.meshObjects.at(10).materialID = 7;
	scene.meshObjects.at(11).materialID = 8;

	//scene.addHDRI(Vector3(0.5));
	scene.addHDRI("C:\\Users\\Kike\\Desktop\\Scenes\\DesertCar\\Export\\goegap_4k.hdr");

	return scene;

}


Scene demoScene2() {

	Scene scene;

	scene.camera = Camera(1000, 1000);
	scene.camera.focalLength = 45;
	scene.camera.position = Vector3(0, 4, -10);

	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\Demo\\CashRegister_01_Diffuse.bmp"));
	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\Demo\\CashRegister_01_Metallic.bmp"));
	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\Demo\\CashRegister_01_Roughness.bmp"));

	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\Demo\\cash_register.obj"));
	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\Models\\pikachubuffed.obj"));

	scene.addMaterial(Material());
	scene.addMaterial(Material());

	scene.materials.at(0).albedoTextureID = 0;
	scene.materials.at(0).roughnessTextureID = 2;
	scene.materials.at(0).metallicTextureID = 1;
	scene.materials.at(0).clearcoat = 0;

	scene.materials.at(1).roughness = 0;
	scene.materials.at(1).metallic = 1;

	scene.meshObjects.at(0).materialID = 0;
	scene.meshObjects.at(1).moveAbsolute(Vector3(7, 3, 20));

	scene.addHDRI("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\Demo\\church.hdr");

	//printHDRISampling(scene.hdri, 10000);

	return scene;

}

Scene coche() {

	Scene scene;

	scene.camera = Camera(1280, 720);
	scene.camera.focalLength = 40;
	scene.camera.position = 0.33 * Vector3(18, 91, -485);
	scene.camera.focusDistance = 200;
	scene.camera.aperture = 12;


	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\EscenaCoche\\carroceria.obj"));
	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\EscenaCoche\\cromado.obj"));
	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\EscenaCoche\\telanegra.obj"));
	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\EscenaCoche\\carretera.obj"));

	scene.addMaterial(Material());
	scene.addMaterial(Material());
	scene.addMaterial(Material());
	scene.addMaterial(Material());

	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\EscenaCoche\\carretera_albedo.bmp"));
	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\EscenaCoche\\carretera_roughness.bmp"));

	scene.materials.at(0).albedo = Vector3(0.6, 0.05, 0.05);
	scene.materials.at(0).metallic = 0;
	scene.materials.at(0).roughness = 0;

	scene.materials.at(1).albedo = Vector3(0.9);
	scene.materials.at(1).metallic = 1;
	scene.materials.at(1).roughness = 0;

	scene.materials.at(2).albedo = Vector3(0.05);
	scene.materials.at(2).metallic = 0;
	scene.materials.at(2).roughness = 0.9;

	scene.materials.at(3).albedoTextureID = 0;
	scene.materials.at(3).roughnessTextureID = 1;

	scene.meshObjects.at(0).materialID = 0;
	scene.meshObjects.at(1).materialID = 1;
	scene.meshObjects.at(2).materialID = 2;
	scene.meshObjects.at(3).materialID = 3;

	scene.addHDRI("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\Demo\\landscape.hdr");

	return scene;

}

struct RenderParameters {

	unsigned int width, height;
	unsigned int sampleTarget;

	float exposure;

	RenderParameters(unsigned int width, unsigned int height, unsigned int sampleTarget, float exposure) : width(width), height(height) , sampleTarget(sampleTarget) , exposure(exposure) {};
	RenderParameters() : width(1280), height(720), sampleTarget(100), exposure(1) {};
};

struct RenderData {

	RenderParameters pars;

	float* rawPixelBuffer;
	unsigned char* beautyBuffer;

	size_t freeMemory = 0;
	size_t totalMemory = 0;

	int pathCount = 0;
	int samples = 0;

	std::chrono::steady_clock::time_point startTime;

	RenderData() {};
};

int startRender(RenderData& data, Scene &scene) {

	RenderParameters pars = data.pars;

	data.rawPixelBuffer = new float[pars.width * pars.height * 4];
	data.beautyBuffer = new unsigned char[pars.width * pars.height * 4];

	memset(data.rawPixelBuffer, 0, pars.width * pars.height * 4 * sizeof(float));
	memset(data.beautyBuffer, 0, pars.width * pars.height * 4 * sizeof(unsigned char));

	cudaError_t cudaStatus = renderSetup(&scene);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "renderCuda failed!");
		return -1;
	}

	t = std::thread(renderCuda, &scene, pars.sampleTarget);

	data.startTime = std::chrono::high_resolution_clock::now();

	return 1;
}

int getRenderData(RenderData& data) {

	int width = data.pars.width;
	int height = data.pars.height;

	int* pathCountBuffer = new int[width * height];

	getBuffer(data.rawPixelBuffer, pathCountBuffer, width * height);
	cudaMemGetInfo(&data.freeMemory, &data.totalMemory);
	flipY(data.rawPixelBuffer,width, height);
	applyExposure(data.rawPixelBuffer, width, height, data.pars.exposure);
	clampPixels(data.rawPixelBuffer, width, height);
	applysRGB(data.rawPixelBuffer, width, height);
	HDRtoLDR(data.rawPixelBuffer, data.beautyBuffer, width, height);

	data.samples = getSamples();
	data.pathCount = 0;

	for (int i = 0; i < width * height; i++)
		data.pathCount += pathCountBuffer[i];

	return 1;
}

int main() {

	Scene scene = clockScene();

	RenderData data;

	data.pars = RenderParameters(scene.camera.xRes, scene.camera.yRes, 100000, 1.0);

	sf::RenderWindow window(sf::VideoMode(data.pars.width, data.pars.height, 32), "Render Window");

	sf::Image image;
	sf::Texture texture;
	sf::Sprite sprite;
	sf::Font font;
	sf::Text text;

	image.create(data.pars.width, data.pars.height);
	texture.loadFromImage(image);
	sprite.setTexture(texture);

	if (!font.loadFromFile("arial.ttf"));
	text.setFont(font);
	text.setCharacterSize(24);
	text.setFillColor(sf::Color::Red);

	startRender(data, scene);

	while (window.isOpen()) {

		sf::Event event;

		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();
		}

		getRenderData(data);

		auto t2 = std::chrono::high_resolution_clock::now();

		auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - data.startTime);

		printf("\rkPaths/s: %f, %fGB of a total of %fGB used, %d/%d samples. %f seconds running, %d total paths",
			((float)data.pathCount / (float)ms_int.count()),
			(float)(data.totalMemory - data.freeMemory) / (1024*1024*1024),
			(float)data.totalMemory / (1024 * 1024 * 1024),
			data.samples,
			data.pars.sampleTarget,
			((float)(ms_int).count())/1000,
			data.pathCount);

		text.setString(std::to_string(data.samples));
		texture.update(data.beautyBuffer);
		window.clear();
		window.draw(sprite);
		window.draw(text);
		window.display();

		std::this_thread::sleep_for(std::chrono::milliseconds(UPDATE_INTERVAL));
	}

	t.join();

	cudaDeviceReset();

	window.close();
}