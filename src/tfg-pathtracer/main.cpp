#include <SFML/Graphics.hpp>
#include <iostream>
#include <Windows.h>
#include "Camera.h"
#include "Sphere.h"
#include "Ray.h"
#include "kernel.h"
#include "Scene.h"
#include <thread>         
#include <chrono>     
#include "Texture.h"
#include "ObjLoader.h"
#include "BVH.h"

Scene disneyMaterialBenchmarkScene() {

	Scene scene;

	scene.addHDRI("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\Models\\church.hdr");

	float sphereRadius = 0.4;
	float separationMult = 1;

	int materialCount = 0;

	scene.camera = Camera(1000, 1000);
	scene.camera.position = Vector3(0, 0, -12);
	scene.camera.focalLength = 35;

	for (int i = 0; i < 10; i++) {

		Material material;
		Sphere sphere(sphereRadius);

		material.albedo = Vector3(1, 1, 1);
		material.subsurface = ((float)i) / 10;

		sphere.position = Vector3((i - 5) * separationMult, separationMult * 5, 0);
		sphere.materialID = materialCount++;

		scene.addMaterial(material);
		scene.addSphere(sphere);
	}


	for (int i = 0; i < 10; i++) {

		Material material;
		Sphere sphere(sphereRadius);

		material.albedo = Vector3(1, 0.75, 0);
		material.roughness = 0;
		material.metallic = ((float)i) / 10;

		sphere.position = Vector3((i - 5) * separationMult, separationMult * 4, 0);
		sphere.materialID = materialCount++;

		scene.addMaterial(material);
		scene.addSphere(sphere);
	}


	return scene;
}

Scene demoScene1() {

	Scene scene;

	scene.camera = Camera(1000, 1000);
	scene.camera.focalLength = 20;
	scene.camera.position = Vector3(0, 0, -8);

	scene.addSphere(Sphere(1));
	scene.addSphere(Sphere(1));
	scene.addSphere(Sphere(1));

	scene.spheres.at(0).position = Vector3(-3, 0, 0);
	scene.spheres.at(1).position = Vector3(0, 0, 0);
	scene.spheres.at(2).position = Vector3(3, 0, 0);

	scene.addMaterial(Material());
	scene.addMaterial(Material());
	scene.addMaterial(Material());

	scene.materials.at(0).roughness = 1;
	scene.materials.at(0).metallic = 0;
	scene.materials.at(0).clearcoat = 0;
	scene.materials.at(0).albedo = Vector3(1, 1, 1);

	scene.materials.at(1).roughness = 0;
	scene.materials.at(1).metallic = 1;
	scene.materials.at(1).clearcoat = 0;
	scene.materials.at(1).albedo = Vector3(1, 1, 1);

	scene.materials.at(2).roughness = 0;
	scene.materials.at(2).metallic = 0;
	scene.materials.at(2).clearcoat = 1;
	scene.materials.at(2).albedo = Vector3(0.1, 1, 1);

	scene.spheres.at(0).materialID = 0;
	scene.spheres.at(1).materialID = 1;
	scene.spheres.at(2).materialID = 2;

	scene.addHDRI("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\Demo\\landscape.hdr");

	scene.hdri.texture.xOffset = -0.25;

	return scene;
}

Scene demoScene2() {

	Scene scene;

	scene.camera = Camera(1000, 1000);
	scene.camera.focalLength = 20;
	scene.camera.position = Vector3(0, 2, -8);

	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\Demo\\kitty_diffuse.bmp"));

	scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\Demo\\kitty.obj"));
	scene.addSphere(1);

	scene.spheres.at(0).position = Vector3(-3, 0, 0);

	scene.addMaterial(Material());
	scene.addMaterial(Material());

	scene.materials.at(0).albedoTextureID = 0;
	scene.materials.at(0).roughness = 0;
	scene.materials.at(0).metallic = 0;
	scene.materials.at(0).clearcoat = 1;

	scene.materials.at(1).roughness = 0;
	scene.materials.at(1).metallic = 1;

	scene.meshObjects.at(0).materialID = 0;
	scene.spheres.at(0).materialID = 1;

	scene.addHDRI("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\Demo\\church.hdr");

	scene.hdri.texture.xOffset = -0.25;

	return scene;

}

Scene testScene() {

		

	Scene scene;

	scene.camera = Camera(1000, 1000);
	scene.camera.focalLength = 20;
	scene.camera.position = Vector3(0, 0, -10);
	scene.hdri.texture.xOffset = 0.20;

	PointLight pl = PointLight();
	pl.position = Vector3(4, 0, 0);
	pl.radiance = Vector3(10, 0, 0);

	PointLight pl1 = PointLight();
	pl1.position = Vector3(-4, 0, 0);
	pl1.radiance = Vector3(0, 10, 0);

	scene.addPointLight(pl);
	scene.addPointLight(pl1);

	//scene.addMeshObject(ObjLoader::loadObj("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\models\\carrolow.obj"));

	scene.addSphere(Sphere(2));
	scene.spheres.at(0).position = Vector3(0, 0, 0);

	scene.addMaterial(Material());

	scene.materials.at(0).roughness = 0;
	scene.materials.at(0).metallic = 1;
	scene.materials.at(0).clearcoat = 0;
	scene.materials.at(0).clearcoatGloss = 0;
	scene.materials.at(0).albedo = Vector3(1,1,1);

	scene.hdri.texture.color = Vector3(0.5);

	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\Metal007_1K_Color.bmp"));
	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\Metal007_1K_Roughness.bmp"));
	scene.addTexture(Texture("C:\\Users\\Kike\\Desktop\\Metal007_1K_Metalness.bmp"));

	scene.materials.at(0).albedoTextureID = 0;
	scene.materials.at(0).roughnessTextureID = 1;
	scene.materials.at(0).metallicTextureID = 2;

	scene.addHDRI("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\Models\\church.hdr");

	scene.hdri.texture.xOffset = 0.25;



	return scene;
}

int main() {

	Scene* scene = new Scene();
	*scene = demoScene2();

	float* pixelBuffer = new float[scene->camera.xRes * scene->camera.yRes * 4];
	sf::Uint8* pixelBuffer_8 = new sf::Uint8[scene->camera.xRes * scene->camera.yRes * 4];



	for (int i = 0; i < scene->camera.xRes * scene->camera.yRes; i++) {

		pixelBuffer[4 * i + 0] = 0;
		pixelBuffer[4 * i + 1] = 0;
		pixelBuffer[4 * i + 2] = 0;
		pixelBuffer[4 * i + 3] = 0;

	}

	sf::RenderWindow window(sf::VideoMode(scene->camera.xRes, scene->camera.yRes, 32), "Render Window");

	cudaError_t cudaStatus = renderSetup(scene);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "renderCuda failed!");
		return 1;
	}

	sf::Image image;
	image.create(scene->camera.xRes, scene->camera.yRes);

	sf::Texture texture;
	texture.loadFromImage(image);

	sf::Sprite sprite;
	sprite.setTexture(texture);

	sf::Font MyFont;

	sf::Font font;
	if (!font.loadFromFile("arial.ttf"));

	sf::Text text;

	text.setFont(font);
	text.setCharacterSize(24);
	text.setFillColor(sf::Color::Red);

	std::thread t1(renderCuda, scene);

	while (window.isOpen()) {

		std::this_thread::sleep_for(std::chrono::seconds(1));

		sf::Event event;

		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();
		}

		getBuffer(pixelBuffer, scene->camera.xRes * scene->camera.yRes * 4);

		for (int x = 0; x < scene->camera.xRes; x++) {
			for (int y = 0; y < scene->camera.yRes; y++) {
				if ((pixelBuffer[4*(scene->camera.xRes * y + x) + 0]) > 1)  (pixelBuffer[4*(scene->camera.xRes * y + x) + 0]) = 1;
				if ((pixelBuffer[4*(scene->camera.xRes * y + x) + 1]) > 1)  (pixelBuffer[4*(scene->camera.xRes * y + x) + 1]) = 1;
				if ((pixelBuffer[4*(scene->camera.xRes * y + x) + 2]) > 1)  (pixelBuffer[4*(scene->camera.xRes * y + x) + 2]) = 1;

				if ((pixelBuffer[4*(scene->camera.xRes * y + x) + 0]) < 0)  (pixelBuffer[4*(scene->camera.xRes * y + x) + 0]) = 0;
				if ((pixelBuffer[4*(scene->camera.xRes * y + x) + 1]) < 0)  (pixelBuffer[4*(scene->camera.xRes * y + x) + 1]) = 0;
				if ((pixelBuffer[4*(scene->camera.xRes * y + x) + 2]) < 0)  (pixelBuffer[4*(scene->camera.xRes * y + x) + 2]) = 0;

				bool sRGB = true;

				if (sRGB) {
					pixelBuffer_8[4 * (scene->camera.xRes * (scene->camera.yRes - y - 1) + (x)) + 0] = pow((pixelBuffer[4*(scene->camera.xRes * y + x) + 0]), 1.0/2.2) * 255.0;
					pixelBuffer_8[4 * (scene->camera.xRes * (scene->camera.yRes - y - 1) + (x)) + 1] = pow((pixelBuffer[4*(scene->camera.xRes * y + x) + 1]), 1.0/2.2) * 255.0;
					pixelBuffer_8[4 * (scene->camera.xRes * (scene->camera.yRes - y - 1) + (x)) + 2] = pow((pixelBuffer[4*(scene->camera.xRes * y + x) + 2]), 1.0/2.2) * 255.0;
				}
				else {

					pixelBuffer_8[4 * (scene->camera.xRes * (scene->camera.yRes - y - 1) + (x)) + 0] = pixelBuffer[4 * (scene->camera.xRes * y + x) + 0] * 255.0;//pow((pixelBuffer[4*(scene->camera.xRes * y + x) + 0]), 1.0/2.2) * 255.0;
					pixelBuffer_8[4 * (scene->camera.xRes * (scene->camera.yRes - y - 1) + (x)) + 1] = pixelBuffer[4 * (scene->camera.xRes * y + x) + 1] * 255.0;//pow((pixelBuffer[4*(scene->camera.xRes * y + x) + 1]), 1.0/2.2) * 255.0;
					pixelBuffer_8[4 * (scene->camera.xRes * (scene->camera.yRes - y - 1) + (x)) + 2] = pixelBuffer[4 * (scene->camera.xRes * y + x) + 2] * 255.0;//pow((pixelBuffer[4*(scene->camera.xRes * y + x) + 2]), 1.0/2.2) * 255.0;
				}

				pixelBuffer_8[4*(scene->camera.xRes * (scene->camera.yRes - y - 1) + (x)) + 3] = 255;
			}
		}


		text.setString(std::to_string(getSamples()));
		texture.update(pixelBuffer_8);
		window.clear();
		window.draw(sprite);
		window.draw(text);
		window.display();
	}

	delete(scene);

	cudaDeviceReset();

}