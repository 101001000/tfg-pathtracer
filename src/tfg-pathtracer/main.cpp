#include <SFML/Graphics.hpp>
#include <thread>         
#include <chrono>    
#include <iostream>
#include <Windows.h>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

#include "Camera.hpp"
#include "Ray.hpp"
#include "kernel.h"
#include "Scene.hpp"
#include "Texture.hpp"
#include "PostProcessing.h"
#include "BVH.hpp"
#include "BMP.hpp"
#include "Definitions.h"


std::thread t;


struct RenderParameters {

	unsigned int width, height;
	unsigned int sampleTarget;

	RenderParameters(unsigned int width, unsigned int height, unsigned int sampleTarget) : width(width), height(height) , sampleTarget(sampleTarget) {};
	RenderParameters() : width(1280), height(720), sampleTarget(100) {};
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

void startRender(RenderData& data, Scene &scene) {

	RenderParameters pars = data.pars;

	data.rawPixelBuffer = new float[pars.width * pars.height * 4];
	data.beautyBuffer = new unsigned char[pars.width * pars.height * 4];

	memset(data.rawPixelBuffer, 0, pars.width * pars.height * 4 * sizeof(float));
	memset(data.beautyBuffer, 0, pars.width * pars.height * 4 * sizeof(unsigned char));

	cudaError_t cudaStatus = renderSetup(&scene);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "renderCuda failed!");
		return;
	}

	t = std::thread(renderCuda, &scene, pars.sampleTarget);

	data.startTime = std::chrono::high_resolution_clock::now();
}

void getRenderData(RenderData& data) {

	int width = data.pars.width;
	int height = data.pars.height;

	int* pathCountBuffer = new int[width * height];

	getBuffer(data.rawPixelBuffer, pathCountBuffer, width * height);
	cudaMemGetInfo(&data.freeMemory, &data.totalMemory);
	flipY(data.rawPixelBuffer,width, height);
	//applyExposure(data.rawPixelBuffer, width, height, data.pars.exposure);
	clampPixels(data.rawPixelBuffer, width, height);
	applysRGB(data.rawPixelBuffer, width, height);
	HDRtoLDR(data.rawPixelBuffer, data.beautyBuffer, width, height);

	data.samples = getSamples();
	data.pathCount = 0;

	for (int i = 0; i < width * height; i++)
		data.pathCount += pathCountBuffer[i];
}

int main(int argc, char* argv[]) {

	//Scene scene = Scene::sceneBuilder(std::string("..\\..\\..\\..\\Scenes\\Clock\\"));

	Scene scene = Scene::sceneBuilder(std::string("C:\\Users\\Kike\\Desktop\\Uni\\TFG\\Scenes\\StandfordDragon\\"));

	RenderData data;

	data.pars = RenderParameters(scene.camera.xRes, scene.camera.yRes, 1000);

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

		if (data.samples >= data.pars.sampleTarget) {
			saveBMP("render.bmp", data.pars.width, data.pars.height, data.beautyBuffer);
			break;
		}

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