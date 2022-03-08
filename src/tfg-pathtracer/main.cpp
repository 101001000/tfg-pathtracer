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


#ifdef _WIN32
#include <Windows.h>
bool keypress(){
	if (GetKeyState('A') & 0x8000)
		return true;
	return false;
}
#endif

std::thread t;
std::thread denoise_thread;

OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);

void startRender(RenderData& data, Scene& scene) {

	RenderParameters pars = data.pars;

	for (int i = 0; i < PASSES_COUNT; i++) {
		if (pars.passes[i]) {
			data.passes[i] = new float[pars.width * pars.height * 4];
			memset(data.passes[i], 0, pars.width * pars.height * 4 * sizeof(float));
		}
	}

	renderSetup(&scene);

	t = std::thread(renderCuda, &scene, pars.sampleTarget);

	data.startTime = std::chrono::high_resolution_clock::now();
}

void denoise(RenderData data, bool* terminate) {
	
	int width = data.pars.width;
	int height = data.pars.height;

	//TODO add stopping condition
	while (true) {

		printf("Denoising...\n");

		// Create a filter for denoising a beauty (color) image using optional auxiliary images too
		OIDNFilter filter = oidnNewFilter(device, "RT"); // generic ray tracing filter
		oidnSetSharedFilterImage(filter, "color", data.passes[BEAUTY], OIDN_FORMAT_FLOAT3, width, height, 0, sizeof(float) * 4, 0); // beauty
		oidnSetFilter1b(filter, "hdr", true); // beauty image is HDR
		oidnSetSharedFilterImage(filter, "output", data.passes[DENOISE], OIDN_FORMAT_FLOAT3, width, height, 0, sizeof(float) * 4, 0); // denoised beauty

		oidnCommitFilter(filter);
		oidnExecuteFilter(filter);

		const char* errorMessage;
		if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
			printf("Error: %s\n", errorMessage);

		// Cleanup
		oidnReleaseFilter(filter);
	}

}

void getRenderData(RenderData& data) {

	int width = data.pars.width;
	int height = data.pars.height;

	int* pathCountBuffer = new int[width * height];

	cudaMemGetInfo(&data.freeMemory, &data.totalMemory);
	getBuffers(data, pathCountBuffer, width * height);

	clampPixels(data.passes[BEAUTY], width, height);
	applysRGB(data.passes[BEAUTY], width, height);

	data.samples = getSamples();
	data.pathCount = 0;

	for (int i = 0; i < width * height; i++)
		data.pathCount += pathCountBuffer[i];

	delete(pathCountBuffer);
}

int main(int argc, char* argv[])
{
	oidnCommitDevice(device);

	bool saved = false;

	Scene scene = loadScene(std::string(argv[1]));
	printf("%s\n", argv[1]);
	RenderData data;
	data.pars = RenderParameters(scene.camera.xRes, scene.camera.yRes, atoi(argv[2]));
	startRender(data, scene);

	Window window(1920, 1080);

	window.init();

	int currentPass = 0;
	bool terminateDenoise = false;

	denoise_thread = std::thread(denoise, data, &terminateDenoise);

	while (!glfwWindowShouldClose(window.window))
	{
		if (keypress()) {
			currentPass++;
			currentPass %= PASSES_COUNT;
		}

		getRenderData(data);
				
		PixelBuffer pb;
		pb.width = data.pars.width;
		pb.height = data.pars.height;

		pb.channels = 4;
		pb.data = data.passes[currentPass];

		window.previewBuffer = pb;

		if (data.samples >= data.pars.sampleTarget - 1 && !saved) {

			unsigned char* saveBuffer = new unsigned char[data.pars.width * data.pars.height * 4];

			for (int i = 0; i < data.pars.width * data.pars.height * 4; i++) {
				saveBuffer[i] = pb.data[i]*255;
			}

			stbi_write_png(argv[3], data.pars.width, data.pars.height, 4, saveBuffer, data.pars.width * 4);
			saved = true;
			delete(saveBuffer);
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

	terminateDenoise = true;
	window.stop();

	oidnReleaseDevice(device);

	denoise_thread.join();
	t.join();
	cudaDeviceReset();
	return 0;
}