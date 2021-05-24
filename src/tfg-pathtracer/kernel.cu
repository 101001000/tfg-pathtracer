#include <iostream>
#include <curand_kernel.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"
#include "Sphere.h"
#include "Camera.h"
#include "Scene.h"
#include "Material.h"
#include "Hit.h"
#include "Disney.h"
#include "PointLight.h"
#include "BVH.h"
#include "HDRI.h"
#include "Math.h"

#define THREADSIZE 16
#define MAXBOUNCES 5

#define USEBVH true
#define HDRIIS true

struct dev_Scene {

    Camera* camera;

    unsigned int sphereCount;
    unsigned int meshObjectCount;
    unsigned int materialCount;
    unsigned int textureCount;
    unsigned int triCount;
    unsigned int pointLightCount;

    PointLight* pointLights;
    Sphere* spheres;
    MeshObject* meshObjects;
    Material* materials;
    Texture* textures;
    Tri* tris;
    BVH* bvh;
    HDRI* hdri;

};

//TODO HACER ESTO CON MEMORIA DINÁMICA PARA ELIMINAR EL MÁXIMO DE 1920*1080

__device__ float dev_buffer[1920 * 1080 * 4]; 

// How many samples per pixels has been calculated. 
__device__ unsigned int dev_samples[1920 * 1080];
__device__ unsigned int dev_pathcount[1920 * 1080];
__device__ dev_Scene* dev_scene_g;
__device__ curandState* d_rand_state_g;

cudaStream_t kernelStream, bufferStream;

long textureMemory = 0;
long geometryMemory = 0;



__device__ void createHitData(Material* material, HitData& hitdata, float u, float v, Vector3 N) {

    Vector3 tangent, bitangent;

    createBasis(N, tangent, bitangent);

    if (material->albedoTextureID < 0) {
        hitdata.albedo = material->albedo;
    }
    else {
        hitdata.albedo = dev_scene_g->textures[material->albedoTextureID].getValueBilinear(u, v);
    }

    if (material->emissionTextureID < 0) {
        hitdata.emission = material->emission;
    }
    else {
        hitdata.emission = dev_scene_g->textures[material->emissionTextureID].getValueBilinear(u, v);
    }

    if (material->roughnessTextureID < 0) {
        hitdata.roughness = material->roughness;
    }
    else {
        hitdata.roughness = dev_scene_g->textures[material->roughnessTextureID].getValueBilinear(u, v).x;
    }

    if (material->metallicTextureID < 0) {
        hitdata.metallic = material->metallic;
    }
    else {
        hitdata.metallic = dev_scene_g->textures[material->metallicTextureID].getValueBilinear(u, v).x;
    }

    hitdata.clearcoatGloss = material->clearcoatGloss;
    hitdata.clearcoat = material->clearcoat;
    hitdata.anisotropic = material->anisotropic;
    hitdata.eta = material->eta;
    hitdata.transmission = material->transmission;
    hitdata.specular = material->specular;
    hitdata.specularTint = material->specularTint;
    hitdata.sheenTint = material->sheenTint;
    hitdata.subsurface = material->subsurface;
    hitdata.sheen = material->sheen;

    hitdata.normal = N;
    hitdata.tangent = tangent;
    hitdata.bitangent = bitangent;
}

__global__ void setupKernel() {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int idx = (dev_scene_g->camera->xRes * y + x);

    if ((x >= dev_scene_g->camera->xRes) || (y >= dev_scene_g->camera->yRes)) return;

    dev_samples[idx] = 0;
    dev_pathcount[idx] = 0;

    dev_buffer[4 * idx + 0] = 0;
    dev_buffer[4 * idx + 1] = 0;
    dev_buffer[4 * idx + 2] = 0;
    dev_buffer[4 * idx + 3] = 1;

    //Inicialización rápida de curand, se pierden propiedades matemáticas o algo así
    curand_init(idx, 0, 0, &d_rand_state_g[idx]);
    
    if (x == 0 && y == 0) {

        int triSum = 0;

        for (int i = 0; i < dev_scene_g->meshObjectCount; i++) {
            dev_scene_g->meshObjects[i].tris += triSum;
            triSum += dev_scene_g->meshObjects[i].triCount;
        }
    }
}

__device__ Hit throwRay(Ray ray, dev_Scene* scene) {

    Hit nearestHit = Hit();
        
    for (int j = 0; j < scene->sphereCount; j++) {

        Hit hit = Hit();

        if (scene->spheres[j].hit(ray, hit)) {

            if (!nearestHit.valid) nearestHit = hit;

            if ((hit.position - ray.origin).length() < (nearestHit.position - ray.origin).length()) {
                nearestHit = hit;
            }
        }
    }

    if (USEBVH) {

        scene->bvh->transverse(ray, nearestHit);

    } else {

        for (int j = 0; j < scene->meshObjectCount; j++) {

            Hit hit = Hit();

            if (scene->meshObjects[j].hit(ray, hit)) {

                if (!nearestHit.valid) {
                    nearestHit = hit;
                }

                if ((hit.position - ray.origin).length() < (nearestHit.position - ray.origin).length()) {
                    nearestHit = hit;
                }
            }
        }
    }
    
    return nearestHit;
}

//TODO limpiar los argumentos de esta función
__device__ Vector3 directLight(Ray ray, HitData hitdata, dev_Scene* scene, Vector3 point, float& pdf, float r1, Vector3& newDir) {

    //TODO falta por dios comprobar colisiones

    // Sampling pointLights

    // We chose a random pointLight to sample from.
    PointLight light = scene->pointLights[(int)(scene->pointLightCount * r1)];

    pdf = scene->pointLightCount;

    newDir = (light.position - point).normalized();

    float dist = (light.position - point).length();

    return (light.radiance / (dist * dist));

}

// Sampling HDRI
// The main idea is to get a random point of the HDRI, weighted by their importance and then get the direction from the center to that point as if that pixel
// would be in a sphere of infinite radius.
__device__ Vector3 hdriLight(Ray ray, dev_Scene* scene, Vector3 point, HitData hitdata, float r1, float r2, float r3, float& pdf) {

    if (!HDRIIS) {

        Vector3 newDir = uniformSampleSphere(r1, r2).normalized();

        float u, v;

        Texture::sphericalMapping(Vector3(), -1 * newDir, 1, u, v);

        Ray shadowRay(point + newDir * 0.001, newDir);

        Hit shadowHit = throwRay(shadowRay, scene);

        if (shadowHit.valid) return Vector3();

        Vector3 hdriValue = scene->hdri->texture.getValueBilinear(u, v);

        Vector3 brdfDisney = DisneyEval(ray, hitdata, newDir);

        return brdfDisney * abs(Vector3::dot(newDir, hitdata.normal)) * hdriValue / (1.0 / (2.0 * PI));
    }
    else {

        Vector3 textCoordinate = scene->hdri->sample(r1);

        //float nu = limitUV((float)textCoordinate.x / (float)scene->hdri->texture.width) + ((r2 - 0.5) / (float)scene->hdri->texture.width);
        //float nv = limitUV((float)textCoordinate.y / (float)scene->hdri->texture.height) + ((r3 - 0.5) / (float)scene->hdri->texture.height);

        float nu = limitUV((float)textCoordinate.x / (float)scene->hdri->texture.width);
        float nv = limitUV((float)textCoordinate.y / (float)scene->hdri->texture.height);

        Vector3 newDir = -1 * Texture::reverseSphericalMapping(nu, nv).normalized();

        Ray shadowRay(point +  newDir * 0.001, newDir);

        Hit shadowHit = throwRay(shadowRay, scene);

        if (shadowHit.valid) return Vector3();

        float u, v;

        Texture::sphericalMapping(Vector3(), -1 * newDir, 1, u, v);

        Vector3 hdriValue = scene->hdri->texture.getValueBilinear(u, v);

        Vector3 brdfDisney = DisneyEval(ray, hitdata, newDir);

        pdf = scene->hdri->pdf(textCoordinate.x, textCoordinate.y);

        return brdfDisney * abs(Vector3::dot(newDir, hitdata.normal)) * hdriValue / pdf;
    }
}

__global__ void neeRenderKernel(){

    dev_Scene* scene = dev_scene_g;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= scene->camera->xRes) || (y >= scene->camera->yRes)) return;

    // The index for the pixel. Maybe I could look for another indexing function which preserves better the spaciality
    int idx = (scene->camera->xRes * y + x);

    float ix;
    float iy;

    unsigned int sa = dev_samples[idx];

    curandState local_rand_state = d_rand_state_g[idx];

    // Relative coordinates for the point where the first ray will be launched
    float dx = scene->camera->position.x + ((float)x) / ((float)scene->camera->xRes) * scene->camera->sensorWidth * 0.001;
    float dy = scene->camera->position.y + ((float)y) / ((float)scene->camera->yRes) * scene->camera->sensorHeight * 0.001;

    // Absolute coordinates for the point where the first ray will be launched
    float odx = (-scene->camera->sensorWidth / 2) * 0.001 + dx;
    float ody = (-scene->camera->sensorHeight / 2) * 0.001 + dy;

    // Random part of the sampling offset so we get antialasing
    float rx = (1.0 / (float)scene->camera->xRes) * (curand_uniform(&local_rand_state) - 0.5) * scene->camera->sensorWidth * 0.001;
    float ry = (1.0 / (float)scene->camera->yRes) * (curand_uniform(&local_rand_state) - 0.5) * scene->camera->sensorHeight * 0.001;

    // The initial ray is created from the camera position to the point calculated before. No rotation is taken into account.
    Ray ray = Ray(scene->camera->position, Vector3(odx + rx, ody + ry, scene->camera->position.z + scene->camera->focalLength * 0.001) - scene->camera->position);

    float diameter = 0.001 * ((scene->camera->focalLength) / scene->camera->aperture);

    float l = (scene->camera->focusDistance + scene->camera->focalLength * 0.001);

    Vector3 focusPoint = ray.origin + ray.direction * (l/(ray.direction.z));

    uniformCircleSampling(curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), ix, iy);

    Vector3 or = scene->camera->position + diameter * Vector3(ix * 0.5, iy * 0.5, 0);

    ray = Ray(or, focusPoint - or);

    // Accumulated radiance
    Vector3 light = Vector3(0, 0, 0);

    // How much light is lost in the path
    Vector3 reduction = Vector3(1, 1, 1);

    // A ray can bounce a max of MAXBOUNCES. This could be changed with russian roulette path termination method, and that would make
    // the renderer unbiased

    int i = 0;

    for (i = 0; i < MAXBOUNCES; i++) {

        int materialID = 0;

        HitData hitdata;

        Hit nearestHit = throwRay(ray, scene);
        Vector3 cN = nearestHit.normal;

        // FIX BACKFACE NORMALS
        if (Vector3::dot(cN, ray.direction) > 0) cN *= -1;

        if (!nearestHit.valid) {
            float u, v;
            Texture::sphericalMapping(Vector3(), -1 * ray.direction, 1, u, v);
            light += scene->hdri->texture.getValueBilinear(u, v) * reduction;
            break;
        }

        if (nearestHit.type == 0) materialID = scene->spheres[nearestHit.objectID].materialID;
        if (nearestHit.type == 1) materialID = scene->meshObjects[nearestHit.objectID].materialID;

        Material* material = &scene->materials[materialID];

        createHitData(material, hitdata, nearestHit.u, nearestHit.v, cN);

        float r1 = curand_uniform(&local_rand_state);
        float r2 = curand_uniform(&local_rand_state);
        float r3 = curand_uniform(&local_rand_state);
        float r4 = curand_uniform(&local_rand_state);
        float r5 = curand_uniform(&local_rand_state);
        float r6 = curand_uniform(&local_rand_state);
        float r7 = curand_uniform(&local_rand_state);


        Vector3 brdfDir = DisneySample(ray, hitdata, r3, r4, r5);;
        Vector3 brdfDisney = DisneyEval(ray, hitdata, brdfDir);

        float brdfPdf = DisneyPdf(ray, hitdata, brdfDir);

        float hdriPdf;

        Vector3 directLight = hdriLight(ray, scene, nearestHit.position, hitdata, r1, r6, r7, hdriPdf);

        float w1 = hdriPdf / (hdriPdf + brdfPdf);
        float w2 = brdfPdf / (hdriPdf + brdfPdf);

        light += w1 * reduction * directLight;
         
        if (brdfPdf <= 0) break;

        reduction *= (brdfDisney * abs(Vector3::dot(brdfDir, cN))) / brdfPdf;

        ray = Ray(nearestHit.position + brdfDir * 0.001, brdfDir);
    }

    dev_pathcount[idx] += i;

    light = clamp(light, 0, 3);


    if (sa > 0) {
        dev_buffer[4 * idx + 0] *= ((float)sa) / ((float)(sa + 1));
        dev_buffer[4 * idx + 1] *= ((float)sa) / ((float)(sa + 1));
        dev_buffer[4 * idx + 2] *= ((float)sa) / ((float)(sa + 1));
    }

    dev_buffer[4 * idx + 0] += light.x / ((float)sa + 1);
    dev_buffer[4 * idx + 1] += light.y / ((float)sa + 1);
    dev_buffer[4 * idx + 2] += light.z / ((float)sa + 1);

    dev_samples[idx]++;

    d_rand_state_g[idx] = local_rand_state;
}


// TODO: BUSCAR FORMA DE SIMPLIFICAR EL SETUP
void pointLightsSetup(Scene* scene, dev_Scene* dev_scene) {

    unsigned int pointLightCount = scene->pointLightCount();

    PointLight* dev_pointLights;

    cudaMemcpy(&dev_scene->pointLightCount, &pointLightCount, sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_pointLights, sizeof(PointLight) * pointLightCount);
    cudaMemcpy(dev_pointLights, scene->getPointLights(), sizeof(PointLight) * pointLightCount, cudaMemcpyHostToDevice);

    cudaMemcpy(&(dev_scene->pointLights), &(dev_pointLights), sizeof(PointLight*), cudaMemcpyHostToDevice);
}
void materialsSetup(Scene* scene, dev_Scene* dev_scene) {

    unsigned int materialCount = scene->materialCount();

    Material* dev_materials;

    cudaMemcpy(&dev_scene->materialCount, &materialCount, sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_materials, sizeof(Material) * materialCount);

    cudaMemcpy(dev_materials, scene->getMaterials(), sizeof(Material) * materialCount, cudaMemcpyHostToDevice);

    cudaMemcpy(&(dev_scene->materials), &(dev_materials), sizeof(Material*), cudaMemcpyHostToDevice);
}
void spheresSetup(Scene* scene, dev_Scene* dev_scene) {

    unsigned int sphereCount = scene->sphereCount();

    Sphere* spheres = scene->getSpheres();

    Sphere* dev_spheres;

    cudaMemcpy(&dev_scene->sphereCount, &sphereCount, sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_spheres, sizeof(Sphere) * sphereCount);

    cudaMemcpy(dev_spheres, spheres, sizeof(Sphere) * sphereCount, cudaMemcpyHostToDevice);

    cudaMemcpy(&(dev_scene->spheres), &(dev_spheres), sizeof(Sphere*), cudaMemcpyHostToDevice);
}
void texturesSetup(Scene* scene, dev_Scene* dev_scene) {

    unsigned int textureCount = scene->textureCount();

    Texture* textures = scene->getTextures();

    Texture* dev_textures;

    cudaMemcpy(&dev_scene->textureCount, &textureCount, sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_textures, sizeof(Texture) * textureCount);
    textureMemory += sizeof(Texture) * textureCount;

    cudaMemcpy(dev_textures, textures, sizeof(Texture) * textureCount, cudaMemcpyHostToDevice);

    for (int i = 0; i < textureCount; i++) {

        float* textureData;

        cudaMalloc((void**)&textureData, sizeof(float) * textures[i].width * textures[i].height * 3);
        textureMemory += sizeof(float) * textures[i].width * textures[i].height * 3;

        cudaMemcpy(textureData, textures[i].data, sizeof(float) * textures[i].width * textures[i].height * 3, cudaMemcpyHostToDevice);
        cudaMemcpy(&(dev_textures[i].data), &textureData, sizeof(float*), cudaMemcpyHostToDevice);

        printf("Texture %d copied, %dpx x %dpx\n", i, textures[i].width, textures[i].height);
    }

    cudaMemcpy(&(dev_scene->textures), &(dev_textures), sizeof(Texture*), cudaMemcpyHostToDevice);
}
void hdriSetup(Scene* scene, dev_Scene* dev_scene) {

    HDRI* hdri = &scene->hdri;

    HDRI* dev_hdri;

    float* dev_data; 
    float* dev_cdf;

    cudaMalloc((void**)&dev_hdri, sizeof(HDRI));
    cudaMalloc((void**)&dev_data, sizeof(float) * hdri->texture.height * hdri->texture.width * 3);
    cudaMalloc((void**)&dev_cdf, sizeof(float) * hdri->texture.height * hdri->texture.width);

    textureMemory += sizeof(HDRI) + sizeof(float) * hdri->texture.height * hdri->texture.width * 4;

    cudaMemcpy(dev_hdri, hdri, sizeof(HDRI), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data, hdri->texture.data, sizeof(float) * hdri->texture.height * hdri->texture.width * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cdf, hdri->cdf, sizeof(float) * hdri->texture.height * hdri->texture.width, cudaMemcpyHostToDevice);

    cudaMemcpy(&(dev_hdri->texture.data), &(dev_data), sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dev_hdri->cdf), &(dev_cdf), sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dev_scene->hdri), &(dev_hdri), sizeof(float*), cudaMemcpyHostToDevice);
}

cudaError_t renderSetup(Scene* scene) {

    printf("Initializing rendering... \n");

    cudaStreamCreate(&kernelStream);

    unsigned int meshObjectCount = scene->meshObjectCount();
    unsigned int triCount = scene->triCount();

    Camera* camera = scene->getMainCamera();

    MeshObject* meshObjects = scene->getMeshObjects();
    Tri* tris = scene->getTris();
    BVH* bvh = scene->buildBVH();

    Camera* dev_camera;
    MeshObject* dev_meshObjects;
    Tri* dev_tris;
    BVH* dev_bvh;
    int* dev_triIndices;

    float* dev_pixelBuffer;

    dev_Scene* dev_scene;
    cudaMalloc((void**)&dev_scene, sizeof(dev_Scene));

    cudaError_t cudaStatus;

    curandState* d_rand_state;
    cudaMalloc((void**)&d_rand_state, camera->xRes * camera->yRes * sizeof(curandState));

    cudaMemcpy(&dev_scene->meshObjectCount, &meshObjectCount, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(&dev_scene->triCount, &triCount, sizeof(unsigned int), cudaMemcpyHostToDevice);

    printf("Copying data to device\n");

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);

    cudaStatus = cudaMalloc((void**)&dev_camera, sizeof(Camera));
    cudaStatus = cudaMalloc((void**)&dev_meshObjects, sizeof(MeshObject) * meshObjectCount);

    cudaStatus = cudaMalloc((void**)&dev_tris, sizeof(Tri) * triCount);
    cudaStatus = cudaMalloc((void**)&dev_bvh, sizeof(BVH));
    cudaStatus = cudaMalloc((void**)&dev_triIndices, sizeof(int) * triCount);

    geometryMemory += sizeof(MeshObject) * meshObjectCount + sizeof(Tri) * triCount + sizeof(BVH) + sizeof(int) * triCount;

    cudaStatus = cudaMemcpy(dev_meshObjects, meshObjects, sizeof(MeshObject) * meshObjectCount, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_tris, tris, sizeof(Tri) * triCount, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_bvh, bvh, sizeof(BVH), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_triIndices, bvh->triIndices, sizeof(int) * triCount, cudaMemcpyHostToDevice);

    for (int i = 0; i < meshObjectCount; i++) {
        cudaMemcpy(&(dev_meshObjects[i].tris), &dev_tris, sizeof(Tri*), cudaMemcpyHostToDevice);
    }

    //Pointer binding for dev_scene

    cudaStatus = cudaMemcpy(&(dev_scene->meshObjects), &(dev_meshObjects), sizeof(MeshObject*), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(&(dev_scene->camera), &(dev_camera), sizeof(Camera*), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(&(dev_scene->tris), &(dev_tris), sizeof(Tri*), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(&(dev_scene->bvh), &(dev_bvh), sizeof(BVH*), cudaMemcpyHostToDevice);

    cudaStatus = cudaMemcpy(&(dev_bvh->tris), &(dev_tris), sizeof(Tri*), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(&(dev_bvh->triIndices), &(dev_triIndices), sizeof(int*), cudaMemcpyHostToDevice);
    
    cudaStatus = cudaMemcpyToSymbol(dev_scene_g, &dev_scene, sizeof(dev_Scene*));
    cudaStatus = cudaMemcpyToSymbol(d_rand_state_g, &d_rand_state, sizeof(curandState*));

    printf("%dMB of geometry data copied\n", (geometryMemory / (1024 * 1024)));

    pointLightsSetup(scene, dev_scene);
    materialsSetup(scene, dev_scene);
    spheresSetup(scene, dev_scene);
    texturesSetup(scene, dev_scene);
    hdriSetup(scene, dev_scene);

    printf("%dMB of texture data copied\n", (textureMemory / (1024 * 1024)));

    int tx = THREADSIZE;
    int ty = THREADSIZE;

    dim3 blocks(camera->xRes / tx + 1, camera->yRes / ty + 1);
    dim3 threads(tx, ty);

    // Launch a kernel on the GPU with one thread for each element.
    setupKernel << <blocks, threads, 0, kernelStream >> >();

    cudaStatus = cudaStreamSynchronize(kernelStream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

Error:

    return cudaStatus;
}

//TODO quitar variables globales pasando por parámetro el puntero.

void renderCuda(Scene* scene) {

    int tx = THREADSIZE;
    int ty = THREADSIZE;

    dim3 blocks(scene->camera.xRes / tx + 1, scene->camera.yRes / ty + 1);
    dim3 threads(tx, ty);

    while (1) {

        neeRenderKernel << <blocks, threads, 0, kernelStream >> > ();

        cudaError_t cudaStatus = cudaStreamSynchronize(kernelStream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        }
    }

}

cudaError_t getBuffer(float* pixelBuffer, int* pathcountBuffer, int size) {

    cudaStreamCreate(&bufferStream);

    cudaError_t cudaStatus = cudaMemcpyFromSymbolAsync(pixelBuffer, dev_buffer, size * sizeof(float) * 4, 0, cudaMemcpyDeviceToHost, bufferStream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "LOL returned error code %d after launching addKernel!\n", cudaStatus);
    }

    cudaStatus = cudaMemcpyFromSymbolAsync(pathcountBuffer, dev_pathcount, size * sizeof(unsigned int), 0, cudaMemcpyDeviceToHost, bufferStream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "LOL returned error code %d after launching addKernel!\n", cudaStatus);
    }

    return cudaStatus;
}

int getSamples() {

    int buff[5];

    cudaStreamCreate(&bufferStream);

    cudaError_t cudaStatus = cudaMemcpyFromSymbolAsync(buff, dev_samples, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost, bufferStream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "LOL returned error code %d after launching addKernel!\n", cudaStatus);
    }

    return buff[0];
}

__host__ void printHDRISampling(HDRI hdri, int samples) {

    for (int i = 0; i < samples; i++) {
        
        float r = ((float)i) / ((float)samples);

        Vector3 sample = hdri.sample(r);

        printf("%d, %d,", (int)sample.x, (int)sample.y);
    }

    float sum = 0;

    for (int i = 0; i < 1024; i++) {
        for (int j = 0; j < 2048; j++) {
            sum += hdri.pdf(j, i);
        }
    }

    printf("PDF TOTAL %f", sum);
}

__host__ void printBRDFMaterial(Material material, int samples) {

    HitData hitdata;

    hitdata.albedo = material.albedo;
    hitdata.emission = material.emission;
    hitdata.roughness = material.roughness;
    hitdata.metallic = material.metallic;
    hitdata.clearcoatGloss = material.clearcoatGloss;
    hitdata.clearcoat = material.clearcoat;
    hitdata.anisotropic = material.anisotropic;
    hitdata.eta = material.eta;
    hitdata.transmission = material.transmission;
    hitdata.specular = material.specular;
    hitdata.specularTint = material.specularTint;
    hitdata.sheenTint = material.sheenTint;
    hitdata.subsurface = material.subsurface;
    hitdata.sheen = material.sheen;

    hitdata.normal = Vector3(0, 1, 0);
    
    createBasis(hitdata.normal, hitdata.tangent, hitdata.bitangent);

    Vector3 inLight = Vector3(1, -1, 0).normalized();

    for (int i = 0; i < sqrt(samples); i++) {
        
        for (int j = 0; j < sqrt(samples); j++) {

            float cosPhi = 2.0f * ((float)i / (float)sqrt(samples)) - 1.0f;
            float sinPhi = std::sqrt(1.0f - cosPhi * cosPhi);
            float theta = 2 * PI * ((float)j / (float)sqrt(samples));

            float x = sinPhi * std::sinf(theta);
            float y = cosPhi;
            float z = sinPhi * std::cosf(theta);

            Vector3 rndVector = Vector3(x, y, z).normalized();

            Ray r = Ray(Vector3(0), inLight);

            float brdf = DisneyEval(r, hitdata, rndVector).length();

            printf("%f,%f,%f,%f;", rndVector.x, rndVector.y, rndVector.z, brdf);
        }
    }
}

/*
__host__ void printPdfMaterial(Material material) {

    int samples = 1000;

    HitData hitdata;

    hitdata.albedo = material.albedo;
    hitdata.roughness = material.roughness;
    hitdata.metallic = material.metallic;
    hitdata.clearcoatGloss = material.clearcoatGloss;
    hitdata.clearcoat = material.clearcoat;
    hitdata.anisotropic = material.anisotropic;
    hitdata.eta = material.eta;
    hitdata.transmission = material.transmission;
    hitdata.specular = material.specular;
    hitdata.specularTint = material.specularTint;
    hitdata.sheenTint = material.sheenTint;
    hitdata.subsurface = material.subsurface;
    hitdata.sheen = material.sheen;

    Vector3 inLight = Vector3(1, -1, 0).normalized();

    for (int i = 0; i < samples; i++) {

        Vector3 rndVector = Vector3(rand(), rand(), rand()).normalized();

        float pdf = DisneyPdf(Ray(Vector3(), inLight), hitdata, rndVector);

        printf("%f, %f, %f, %f\n", rndVector.x, rndVector.y, rndVector.z, pdf);
    }
}
*/