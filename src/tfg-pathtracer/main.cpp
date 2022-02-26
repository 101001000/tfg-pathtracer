#include <SFML/Graphics.hpp>
#include <thread>         
#include <chrono>    
#include <iostream>
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
#include "SceneLoader.hpp"


std::thread t;

void startRender(RenderData& data, Scene& scene) {

	RenderParameters pars = data.pars;

	for (int i = 0; i < PASSES_COUNT; i++) {
		if (pars.passes[i]) {
			data.buffers[i] = new float[pars.width * pars.height * 4];
			memset(data.buffers[i], 0, pars.width * pars.height * 4 * sizeof(float));
		}
	}

	data.beautyBuffer = new unsigned char[pars.width * pars.height * 4];

	memset(data.beautyBuffer, 0, pars.width * pars.height * 4 * sizeof(unsigned char));

	renderSetup(&scene);

	t = std::thread(renderCuda, &scene, pars.sampleTarget);

	data.startTime = std::chrono::high_resolution_clock::now();
}

void getRenderData(RenderData& data) {

	int width = data.pars.width;
	int height = data.pars.height;

	int* pathCountBuffer = new int[width * height];

	cudaMemGetInfo(&data.freeMemory, &data.totalMemory);

	getBuffers(data, pathCountBuffer, width * height);

	flipY(data.buffers[BEAUTY], width, height);
	clampPixels(data.buffers[BEAUTY], width, height);
	applysRGB(data.buffers[BEAUTY], width, height);
	HDRtoLDR(data.buffers[BEAUTY], data.beautyBuffer, width, height);

	data.samples = getSamples();
	data.pathCount = 0;

	for (int i = 0; i < width * height; i++)
		data.pathCount += pathCountBuffer[i];

	delete(pathCountBuffer);
}

int main(int argc, char* argv[]) {

	Scene scene = loadScene(std::string(argv[1]));

	printf("%s\n", argv[1]);

	RenderData data;

	data.pars = RenderParameters(scene.camera.xRes, scene.camera.yRes, atoi(argv[2]));

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

		if (data.samples >= data.pars.sampleTarget - 1) {
			saveBMP(argv[3], data.pars.width, data.pars.height, data.beautyBuffer);
			break;
		}

		auto t2 = std::chrono::high_resolution_clock::now();

		auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - data.startTime);

		printf("\rkPaths/s: %f, %fGB of a total of %fGB used, %d/%d samples. %f seconds running, %d total paths",
			((float)data.pathCount / (float)ms_int.count()),
			(float)(data.totalMemory - data.freeMemory) / (1024 * 1024 * 1024),
			(float)data.totalMemory / (1024 * 1024 * 1024),
			data.samples,
			data.pars.sampleTarget,
			((float)(ms_int).count()) / 1000,
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
	return 0;
}