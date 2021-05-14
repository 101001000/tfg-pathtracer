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

Scene cocheRefachero() {

	Scene scene;

	scene.camera = Camera(1280, 960);
	scene.camera.position = Vector3(1.739, -1.5, -8.175);
	scene.camera.focalLength = 40;
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

	//scene.hdri.texture.xOffset = 0.73;

	printHDRISampling(scene.hdri, 10000);

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

void applyExposure(float* pixels, int width, int height, float exposure) {
	for (int i = 0; i < width * height * 4; i++)
		pixels[i] *= exposure;
}

void flipY(float* pixels, int width, int height) {
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height/2; y++) {
			float temp[4];
			memcpy(temp, (void*) &pixels[(x + y * width) * 4], sizeof(float) * 4);
			memcpy((void*) &pixels[(x + y * width) * 4], (void*) &pixels[(x + (height-y-1) * width) * 4], sizeof(float) * 4);
			memcpy((void*)&pixels[(x + (height - y-1) * width) * 4], (void*)temp, sizeof(float) * 4);
		}
	}	
}

void flipX(float* pixels, int width, int height) {
	for (int x = 0; x < width/2; x++) {
		for (int y = 0; y < height; y++) {
			float temp[4];
			memcpy(temp, (void*)&pixels[(x + y * width) * 4], sizeof(float) * 4);
			memcpy((void*)&pixels[(x + y * width) * 4], (void*)&pixels[((width - x-1) + y * width) * 4], sizeof(float) * 4);
			memcpy((void*)&pixels[((width - x - 1) + y * width) * 4], (void*)temp, sizeof(float) * 4);
		}
	}
}

void clampPixels(float* pixels, int width, int height) {
	for (int i = 0; i < width * height * 4; i++)
		pixels[i] = clamp(pixels[i], 0, 1);
}

void reinhardTonemap(float* pixels, int width, int height) {

	for (int i = 0; i < width * height * 4; i++)
		pixels[i] = pixels[i] / (1.0f + pixels[i]);
}

void acesTonemap(float* pixels, int width, int height) {

	float a = 2.51f;
	float b = 0.03f;
	float c = 2.43f;
	float d = 0.59f;
	float e = 0.14f;

	for (int i = 0; i < width * height * 4; i++)
		pixels[i] = clamp((pixels[i] * 0.6 * (a * pixels[i] * 0.6 + b)) / (pixels[i] * 0.6 * (c * pixels[i] * 0.6 + d) + e), 0.0f, 1.0f);

}

void applysRGB(float* pixels, int width, int height) {
	for (int i = 0; i < width * height * 4; i++)
		pixels[i] = pow(pixels[i], 1.0 / 2.2);
}


float gaussianDist(float x, float sigma) {
	return 0.39894 * exp(-0.5 * x * x / (sigma * sigma)) / sigma;
}

void getThreshold(float* pixels, int width, int height, float threshold, float* result) {

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {

			float v = 0;

			v += pixels[(x + y * width) * 4 + 0] / 3.0;
			v += pixels[(x + y * width) * 4 + 1] / 3.0;
			v += pixels[(x + y * width) * 4 + 2] / 3.0;

			if (v < threshold) {
				result[(x + y * width) * 4 + 0] = 0;
				result[(x + y * width) * 4 + 1] = 0;
				result[(x + y * width) * 4 + 2] = 0;
				result[(x + y * width) * 4 + 2] = 0;
			}
			else {
				result[(x + y * width) * 4 + 0] = pixels[(x + y * width) * 4 + 0];
				result[(x + y * width) * 4 + 1] = pixels[(x + y * width) * 4 + 1];
				result[(x + y * width) * 4 + 2] = pixels[(x + y * width) * 4 + 2];
				result[(x + y * width) * 4 + 3] = pixels[(x + y * width) * 4 + 3];
			}
		}
	}

}

void gaussianBlur(float* pixels, int width, int height, int kernelSize, float* result) {

	float* resultTemp = new float[4 * width * height];

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {

			float sum[4] = { 0, 0, 0, 0 };

			for (int xk = 0; xk < kernelSize; xk++) {
				for (int yk = 0; yk < kernelSize; yk++) {

					float t1 = 1.0 / ((2.0 * PI) * ((float)kernelSize * (float)kernelSize));
					float tx = xk - (int)(kernelSize / 2.0);
					float ty = yk - (int)(kernelSize / 2.0);
					float t2 = -((tx * tx) + (ty * ty)) / (2.0 * (float)kernelSize * (float)kernelSize);

					float w = t1 * exp(t2);
					
					int xx = clamp(x + tx, 0, width - 1);
					int yy = clamp(y + ty, 0, height - 1);

					sum[0] += w * pixels[(xx + yy * width) * 4 + 0];
					sum[1] += w * pixels[(xx + yy * width) * 4 + 1];
					sum[2] += w * pixels[(xx + yy * width) * 4 + 2];
				}
			}

			resultTemp[(x + y * width) * 4 + 0] = sum[0];
			resultTemp[(x + y * width) * 4 + 1] = sum[1];
			resultTemp[(x + y * width) * 4 + 2] = sum[2];
			resultTemp[(x + y * width) * 4 + 3] = 1;
		}
	}

	for (int i = 0; i < width * height * 4; i++)
		result[i] = resultTemp[i];

	delete(resultTemp);
}

void basicBlur(float* pixels, int width, int height, float threshold, float power, float radius) {

	int blurSize = radius/2;
	float sd = radius;
	float kernelSize = (blurSize * 2 + 1);

	float* blurMatrix = new float[4 * width * height];
	float* thresholdMatrix = new float[4 * width * height];
	float* kernel = new float[kernelSize * kernelSize];


	for (int x = 0; x < kernelSize; x++) {
		for (int y = 0; y < kernelSize; y++) {

			float t1 = 1.0 / ((2.0 * PI) * (sd * sd));
			float tx = x - (int)(kernelSize / 2.0);
			float ty = y - (int)(kernelSize / 2.0);
			float t2 = -((tx * tx) + (ty * ty)) / (2.0 * sd * sd);

			kernel[(int)(x + y * kernelSize)] = t1 * exp(t2);
		}
	}

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {

			float v = 0;

			v += pixels[(x + y * width) * 4 + 0] / 3.0;
			v += pixels[(x + y * width) * 4 + 1] / 3.0;
			v += pixels[(x + y * width) * 4 + 2] / 3.0;

			if (v < threshold) {
				thresholdMatrix[(x + y * width) * 4 + 0] = 0;
				thresholdMatrix[(x + y * width) * 4 + 1] = 0;
				thresholdMatrix[(x + y * width) * 4 + 2] = 0;
				thresholdMatrix[(x + y * width) * 4 + 2] = 0;
			}
			else {
				thresholdMatrix[(x + y * width) * 4 + 0] = pixels[(x + y * width) * 4 + 0];
				thresholdMatrix[(x + y * width) * 4 + 1] = pixels[(x + y * width) * 4 + 1];
				thresholdMatrix[(x + y * width) * 4 + 2] = pixels[(x + y * width) * 4 + 2];
				thresholdMatrix[(x + y * width) * 4 + 3] = pixels[(x + y * width) * 4 + 3];
			}
		}
	}
	

	for (int x = blurSize; x < width - blurSize; x++) {
		for (int y = blurSize; y < height - blurSize; y++) {

			float v[4] = {0,0,0,0};

			int ii = 0;

			for (int i = x - blurSize; i <= x + blurSize; i++) {

				int jj = 0;

				for (int j = y - blurSize; j <= y + blurSize; j++) {
					v[0] += thresholdMatrix[(i + j * width) * 4 + 0] * kernel[(int)(ii + jj * kernelSize)];
					v[1] += thresholdMatrix[(i + j * width) * 4 + 1] * kernel[(int)(ii + jj * kernelSize)];
					v[2] += thresholdMatrix[(i + j * width) * 4 + 2] * kernel[(int)(ii + jj * kernelSize)];
					v[3] += thresholdMatrix[(i + j * width) * 4 + 3] * kernel[(int)(ii + jj * kernelSize)];
					jj++;
				}
				ii++;
			}
			
			blurMatrix[(x + y * width) * 4 + 0] = v[0];
			blurMatrix[(x + y * width) * 4 + 1] = v[1];
			blurMatrix[(x + y * width) * 4 + 2] = v[2];
			blurMatrix[(x + y * width) * 4 + 3] = v[3];
		}
	}

	for (int i = 0; i < width * height * 4; i++)
		pixels[i] += blurMatrix[i] * power;

}

void downscale(float* pixels, int width, int height, int nWidth, int nHeight, float* result) {

	float rx = width / nWidth;
	float ry = height / nHeight;

	for (int x = 0; x < nWidth; x++) {
		for (int y = 0; y < nHeight; y++) {
			result[4 * (y * nWidth + x) + 0] = pixels[4 * ((int)(y * ry * width) + (int)(x * rx)) + 0];
			result[4 * (y * nWidth + x) + 1] = pixels[4 * ((int)(y * ry * width) + (int)(x * rx)) + 1];
			result[4 * (y * nWidth + x) + 2] = pixels[4 * ((int)(y * ry * width) + (int)(x * rx)) + 2];
			result[4 * (y * nWidth + x) + 3] = 1;
		}
	}
}

void upscale(float* pixels, int width, int height, int nWidth, int nHeight, float* result) {

	float rx = width / nWidth;
	float ry = height / nHeight;

	for (int x = 0; x < nWidth; x++) {
		for (int y = 0; y < nHeight; y++) {
			result[4 * (y * nWidth + x) + 0] = pixels[4 * ((int)(y * ry * width) + (int)(x * rx)) + 0];
			result[4 * (y * nWidth + x) + 1] = pixels[4 * ((int)(y * ry * width) + (int)(x * rx)) + 1];
			result[4 * (y * nWidth + x) + 2] = pixels[4 * ((int)(y * ry * width) + (int)(x * rx)) + 2];
			result[4 * (y * nWidth + x) + 3] = 1;
		}
	}
}

void beautyBloom(float* pixels, int width, int height, float threshold, float power, float radius) {

	int nw = 640;
	int nh = 360;

	float* pixelsDown = new float[nw * nh * 4];

	downscale(pixels, width, height, nw, nh, pixelsDown);
	upscale(pixelsDown, nw, nh, width, height, pixels);

	/*

	float bloomFactors[] = { 1.0, 0.8, 0.6, 0.4, 0.2 };

	int kernelSizes[] = { 3, 5, 7, 9, 11 };

	int nW = width;
	int nH = height;

	for (int i = 0; i < 5; i++) {

		float* pixelsDown = new float[nW*nH*4];
		float* pixelsUp = new float[width * height * 4];
		float* thresholdMatrix = new float[nW * nH * 4];
		float* blurMatrix = new float[nW * nH * 4];

		downscale(pixels, width, height, nW, nH, pixelsDown);
		getThreshold(pixelsDown, nW, nH, threshold, thresholdMatrix);
		gaussianBlur(thresholdMatrix, nW, nH, kernelSizes[i], blurMatrix);
		upscale(pixels, nW, nH, width, height, pixelsUp);

		for (int j = 0; j < width * height; j++) {

			float w = power * lerp(bloomFactors[i], 1.2 - bloomFactors[i], radius);

			pixels[j * 4 + 0] += w * pixelsUp[j * 4 + 0];
			pixels[j * 4 + 1] += w * pixelsUp[j * 4 + 1];
			pixels[j * 4 + 2] += w * pixelsUp[j * 4 + 2];
			pixels[j * 4 + 3] += w * pixelsUp[j * 4 + 3];
		}

		delete(pixelsDown);
		delete(pixelsUp);
		delete(thresholdMatrix);
		delete(blurMatrix);
	}
	*/
}

void HDRtoLDR(float* pixelsIn, sf::Uint8* pixelsOut, int width, int height) {
	for (int i = 0; i < width * height * 4; i++)
		pixelsOut[i] = (sf::Uint8)(pixelsIn[i] * 255);
}

int main() {

	Scene* scene = new Scene();
	*scene = cocheRefachero();

	int width = scene->camera.xRes;
	int height = scene->camera.yRes;

	float exposure = 1;

	float* pixelBuffer = new float[width * height * 4];
	sf::Uint8* pixelBuffer_8 = new sf::Uint8[width * height * 4];

	memset(pixelBuffer, width * height * 4, 0);

	sf::RenderWindow window(sf::VideoMode(width, height, 32), "Render Window");

	cudaError_t cudaStatus = renderSetup(scene);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "renderCuda failed!");
		return 1;
	}

	sf::Image image;
	image.create(width, height);

	sf::Texture texture;
	texture.loadFromImage(image);

	sf::Sprite sprite;
	sprite.setTexture(texture);

	sf::Font font;
	if (!font.loadFromFile("arial.ttf"));

	sf::Text text;

	text.setFont(font);
	text.setCharacterSize(24);
	text.setFillColor(sf::Color::Red);

	std::thread t1(renderCuda, scene);

	while (window.isOpen()) {

		std::this_thread::sleep_for(std::chrono::milliseconds(500));

		sf::Event event;

		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();
		}

		getBuffer(pixelBuffer, width * height * 4);

		flipY(pixelBuffer, width, height);
		//flipX(pixelBuffer, width, height);
		applyExposure(pixelBuffer, width, height, exposure);
		//basicBlur(pixelBuffer, width, height, 0.5, 1, 10);
		//getThreshold(pixelBuffer, width, height, 0.7, pixelBuffer);
		//gaussianBlur(pixelBuffer, width, height, 13, pixelBuffer);
		//beautyBloom(pixelBuffer, width, height, 0.3, 1.6, 1);
		//reinhardTonemap(pixelBuffer, width, height);
		clampPixels(pixelBuffer, width, height);
		applysRGB(pixelBuffer, width, height);
		HDRtoLDR(pixelBuffer, pixelBuffer_8, width, height);

		text.setString(std::to_string(getSamples()));
		texture.update(pixelBuffer_8);
		window.clear();
		window.draw(sprite);
		window.draw(text);
		window.display();
	}

	t1.join();

	cudaDeviceReset();

	window.close();

	delete(scene);



}