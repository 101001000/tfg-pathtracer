#include <thread>         
#include <chrono>    
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <OpenImageDenoise/oidn.hpp>


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Window.hpp"
#include "Camera.hpp"
#include "Ray.hpp"
#include "kernel.h"
#include "Scene.hpp"
#include "Texture.hpp"
#include "PostProcessing.h"
#include "BVH.hpp"
#include "Definitions.h"
#include "SceneLoader.hpp"

/*
#ifdef _WIN32
#include <Windows.h>
void keypress(){
	if (GetKeyState('A') & 0x8000) {
		pass++;
		pass %= PASSES_COUNT;
	}
}
#endif*/

std::thread t;
OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);

void startRender(RenderData& data, Scene& scene) {

	RenderParameters pars = data.pars;

	for (int i = 0; i < PASSES_COUNT; i++) {
		if (pars.passes[i]) {
			data.passes[i] = new float[pars.width * pars.height * 4];
			memset(data.passes[i], 0, pars.width * pars.height * 4 * sizeof(float));
		}
	}

	data.outputBuffer = new unsigned char[pars.width * pars.height * 3];
	data.denoisedBuffer = new float[pars.width * pars.height * 3];

	memset(data.outputBuffer, 0, pars.width * pars.height * 3 * sizeof(unsigned char));

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

	flipY(data.passes[BEAUTY], width, height);
	clampPixels(data.passes[BEAUTY], width, height);
	applysRGB(data.passes[BEAUTY], width, height);

	// Create a filter for denoising a beauty (color) image using optional auxiliary images too
	OIDNFilter filter = oidnNewFilter(device, "RT"); // generic ray tracing filter
	oidnSetSharedFilterImage(filter, "color", data.passes[BEAUTY], OIDN_FORMAT_FLOAT3, data.pars.width, data.pars.height, 0, sizeof(float) * 4, 0); // beauty
	oidnSetFilter1b(filter, "hdr", true); // beauty image is HDR
	oidnSetSharedFilterImage(filter, "output", data.denoisedBuffer, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // denoised beauty

	oidnCommitFilter(filter);
	oidnExecuteFilter(filter);

	const char* errorMessage;
	if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
		printf("Error: %s\n", errorMessage);

	// Cleanup
	oidnReleaseFilter(filter);
	
	HDRtoLDR(data.denoisedBuffer, data.outputBuffer, width, height);

	data.samples = getSamples();
	data.pathCount = 0;

	for (int i = 0; i < width * height; i++)
		data.pathCount += pathCountBuffer[i];

	delete(pathCountBuffer);
}

int main(int argc, char* argv[])
{
	oidnCommitDevice(device);

	Scene scene = loadScene(std::string(argv[1]));
	printf("%s\n", argv[1]);
	RenderData data;
	data.pars = RenderParameters(scene.camera.xRes, scene.camera.yRes, atoi(argv[2]));
	startRender(data, scene);

	Window window(1920, 1080);

	window.init();

	while (!glfwWindowShouldClose(window.window))
	{
		getRenderData(data);
				
		window.outputBuffer = data.outputBuffer;

		if (data.samples >= data.pars.sampleTarget - 1) {
			stbi_write_png(argv[3], data.pars.width, data.pars.height, 3, data.outputBuffer, data.pars.width * 3);
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
		
		window.renderUpdate();

		std::this_thread::sleep_for(std::chrono::milliseconds(UPDATE_INTERVAL));
	}

	window.stop();

	oidnReleaseDevice(device);
	t.join();
	cudaDeviceReset();
	return 0;
}