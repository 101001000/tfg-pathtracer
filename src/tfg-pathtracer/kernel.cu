#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "kernel.h"
#include "Camera.hpp"
#include "Material.hpp"
#include "Hit.hpp"
#include "Disney.hpp"
#include "PointLight.hpp"
#include "BVH.hpp"
#include "HDRI.hpp"
#include "Math.hpp"
#include "Definitions.h"
struct dev_Scene {

    Camera* camera;

    unsigned int sphereCount;
    unsigned int meshObjectCount;
    unsigned int materialCount;
    unsigned int textureCount;
    unsigned int triCount;
    unsigned int pointLightCount;

    PointLight* pointLights;
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
__device__  unsigned int dev_samples[1920 * 1080];
__device__ unsigned int dev_pathcount[1920 * 1080];
__device__ dev_Scene* dev_scene_g;
__device__ curandState* d_rand_state_g;

cudaStream_t kernelStream, bufferStream;

long textureMemory = 0;
long geometryMemory = 0;

__device__ void generateHitData(Material* material, HitData& hitdata, Hit hit) {

    Vector3 tangent, bitangent, normal;

    normal = hit.normal;
    tangent = hit.tangent;
    bitangent = hit.bitangent;

    if (material->albedoTextureID < 0) {
        hitdata.albedo = material->albedo;
    }
    else {
        hitdata.albedo = dev_scene_g->textures[material->albedoTextureID].getValueBilinear(hit.tu, hit.tv);
    }

    if (material->emissionTextureID < 0) {
        hitdata.emission = material->emission;
    }
    else {
        hitdata.emission = dev_scene_g->textures[material->emissionTextureID].getValueBilinear(hit.tu, hit.tv);
    }

    if (material->roughnessTextureID < 0) {
        hitdata.roughness = material->roughness;
    }
    else {
        hitdata.roughness = dev_scene_g->textures[material->roughnessTextureID].getValueBilinear(hit.tu, hit.tv).x;
    }

    if (material->metallicTextureID < 0) {
        hitdata.metallic = material->metallic;
    }
    else {
        hitdata.metallic = dev_scene_g->textures[material->metallicTextureID].getValueBilinear(hit.tu, hit.tv).x;
    }

    if (material->normalTextureID < 0) {
        hitdata.normal = normal;
    }
    else {

        Vector3 ncolor = dev_scene_g->textures[material->normalTextureID].getValueFromUV(hit.tu, hit.tv);

        Vector3 localNormal = (ncolor * 2) - 1;


        //localNormal = Vector3(localNormal.x, 0, 0);

        //localNormal = Vector3(0, 0, 1);

        /*
        Vector3 ws_normal = Vector3(localNormal.x  * tangent.x + localNormal.y  * -bitangent.x + localNormal.z  * normal.x,
                                    localNormal.x  * tangent.y + localNormal.y  * -bitangent.y + localNormal.z  * normal.y,
                                    localNormal.x  * tangent.z + localNormal.y  * -bitangent.z + localNormal.z  * normal.z).normalized();*/

        Vector3 worldNormal = (localNormal.x * tangent - localNormal.y * bitangent + localNormal.z * normal).normalized();

        hitdata.normal = worldNormal;
        //hitdata.albedo = clamp(Vector3(ws_normal.x, ws_normal.z, ws_normal.y), 0 , 1);
    }

    // Convert linear to sRGB
    hitdata.roughness = pow(hitdata.roughness, 2.2f);
    hitdata.metallic = pow(hitdata.metallic, 2.2f);

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

    // Esto se puede hacer con un cudaMemset
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

#ifdef USEBVH:
    scene->bvh->transverse(ray, nearestHit);
    //scene->bvh->transverseAux(ray, nearestHit, scene->bvh->nodes[0]);
#else:
    for (int j = 0; j < scene->meshObjectCount; j++) {

        Hit hit = Hit();

        if (scene->meshObjects[j].hit(ray, hit)) {
            if (!nearestHit.valid)
                nearestHit = hit;

            if ((hit.position - ray.origin).length() < (nearestHit.position - ray.origin).length())
                nearestHit = hit;
        }
    }
#endif    
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

        Vector3 hdriValue = scene->hdri->texture.getValueFromUV(u, v);

        Vector3 brdfDisney = DisneyEval(ray, hitdata, newDir);

        pdf = (1.0 / (2.0 * PI * PI));

        return brdfDisney * abs(Vector3::dot(newDir, hitdata.normal)) * hdriValue / (1.0 / (2.0 * PI * PI));
    }
    else {

        Vector3 textCoordinate = scene->hdri->sample(r1);

        float nu = textCoordinate.x / (float)scene->hdri->texture.width;
        float nv = textCoordinate.y / (float)scene->hdri->texture.height;

        float iu = scene->hdri->texture.inverseTransformUV(nu, nv).x;
        float iv = scene->hdri->texture.inverseTransformUV(nu, nv).y;

        Vector3 newDir = -scene->hdri->texture.reverseSphericalMapping(iu, iv).normalized();

        Ray shadowRay(point + newDir * 0.001, newDir);

        Hit shadowHit = throwRay(shadowRay, scene);

        if (shadowHit.valid) return Vector3();

        Vector3 hdriValue = scene->hdri->texture.getValueFromUV(iu, iv);

        Vector3 brdfDisney = DisneyEval(ray, hitdata, newDir);

        pdf = scene->hdri->pdf(iu * scene->hdri->texture.width, iv * scene->hdri->texture.height);

        return brdfDisney * abs(Vector3::dot(newDir, hitdata.normal)) * hdriValue / pdf;
    }
}

__device__ void calculateCameraRay(int x, int y, Camera& camera, Ray& ray, float r1, float r2, float r3, float r4, float r5) {

    // Relative coordinates for the point where the first ray will be launched
    float dx = camera.position.x + ((float)x) / ((float)camera.xRes) * camera.sensorWidth;
    float dy = camera.position.y + ((float)y) / ((float)camera.yRes) * camera.sensorHeight;

    // Absolute coordinates for the point where the first ray will be launched
    float odx = (-camera.sensorWidth / 2.0) + dx;
    float ody = (-camera.sensorHeight / 2.0) + dy;

    // Random part of the sampling offset so we get antialasing
    float rx = (1.0 / (float)camera.xRes) * (r1 - 0.5) * camera.sensorWidth;
    float ry = (1.0 / (float)camera.yRes) * (r2 - 0.5) * camera.sensorHeight;

    // Sensor point, the point where intersects the ray with the sensor
    float SPx = odx + rx;
    float SPy = ody + ry;
    float SPz = camera.position.z + camera.focalLength;

    // The initial ray is created from the camera position to the sensor point. No rotation is taken into account.
    ray = Ray(camera.position, Vector3(SPx, SPy, SPz) - camera.position);

#if BOKEH

    float rIPx, rIPy;

    // The diameter of the camera iris
    float diameter = camera.focalLength / camera.aperture;

    // Total length from the camera to the focus plane
    float l = camera.focusDistance + camera.focalLength;

    // The point from the initial ray which is actually in focus
    //Vector3 focusPoint = ray.origin + ray.direction * (l / (ray.direction.z));
    // Mala aproximación, encontrar soluición
    Vector3 focusPoint = ray.origin + ray.direction * l;

    // Sampling for the iris of the camera
    uniformCircleSampling(r3, r4, r5, rIPx, rIPy);

    rIPx *= diameter * 0.5;
    rIPy *= diameter * 0.5;

    Vector3 orig = camera.position + Vector3(rIPx, rIPy, 0);

    //Blurred ray
    ray = Ray(orig, focusPoint - orig);

#endif 
}

__device__ void shade(dev_Scene& scene, Ray& ray, HitData& hitdata, Hit& nearestHit, Vector3& newDir, float r1, float r2, float r3, Vector3& hitLight, Vector3& reduction) {

    Vector3 brdfDisney = DisneyEval(ray, hitdata, newDir);

    float brdfPdf = DisneyPdf(ray, hitdata, newDir);

    float hdriPdf;

    Vector3 directLight = hdriLight(ray, &scene, nearestHit.position, hitdata, r1, r2, r3, hdriPdf);

    float w1 = hdriPdf / (hdriPdf + brdfPdf);
    float w2 = brdfPdf / (hdriPdf + brdfPdf);

    if (brdfPdf <= 0) return;

    hitLight = w1 * reduction * directLight;

    reduction *= (brdfDisney * abs(Vector3::dot(newDir, hitdata.normal))) / brdfPdf;

}

__device__ void calculateBounce(Ray& incomingRay, HitData& hitdata, Vector3& bouncedDir, float r1, float r2, float r3) {

    bouncedDir = DisneySample(incomingRay, hitdata, r1, r2, r3);

}

__global__ void neeRenderKernel() {

    dev_Scene* scene = dev_scene_g;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= scene->camera->xRes) || (y >= scene->camera->yRes)) return;

    int idx = (scene->camera->xRes * y + x);

    curandState local_rand_state = d_rand_state_g[idx];

    unsigned int sa = dev_samples[idx];

    Ray ray;

    calculateCameraRay(x, y, *scene->camera, ray, curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), curand_uniform(&local_rand_state));

    // Accumulated radiance
    Vector3 light = Vector3::Zero();

    // How much light is lost in the path
    Vector3 reduction = Vector3::One();

    // A ray can bounce a max of MAXBOUNCES. This could be changed with russian roulette path termination method, and that would make
    // the renderer unbiased

    int i = 0;

    for (i = 0; i < MAXBOUNCES; i++) {

        Vector3 hitLight;
        HitData hitdata;
        Vector3 bouncedDir;

        int materialID = 0;

        Hit nearestHit = throwRay(ray, scene);

        if (!nearestHit.valid) {
            float u, v;
            Texture::sphericalMapping(Vector3(), -1 * ray.direction, 1, u, v);
            light += scene->hdri->texture.getValueBilinear(u, v) * reduction;
            break;
        }

        //if (Vector3::dot(nearestHit.normal, ray.direction) > 0) nearestHit.normal *= -1;

        materialID = scene->meshObjects[nearestHit.objectID].materialID;

        Material* material = &scene->materials[materialID];

        generateHitData(material, hitdata, nearestHit);

        calculateBounce(ray, hitdata, bouncedDir, curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), curand_uniform(&local_rand_state));

        shade(*scene, ray, hitdata, nearestHit, bouncedDir, curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), hitLight, reduction);

        float hdriPdf;

        //Vector3 hLight = hdriLight(ray, scene, nearestHit.position, hitdata, curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), hdriPdf);;

        light += hitLight;

        ray = Ray(nearestHit.position + bouncedDir * 0.001, bouncedDir);
    }

    dev_pathcount[idx] += i;

    light = clamp(light, 0, 10);

    if (!isnan(light.x) && !isnan(light.y) && !isnan(light.z)) {

        if (sa > 0) {
            dev_buffer[4 * idx + 0] *= ((float)sa) / ((float)(sa + 1));
            dev_buffer[4 * idx + 1] *= ((float)sa) / ((float)(sa + 1));
            dev_buffer[4 * idx + 2] *= ((float)sa) / ((float)(sa + 1));
        }

        dev_buffer[4 * idx + 0] += light.x / ((float)sa + 1);
        dev_buffer[4 * idx + 1] += light.y / ((float)sa + 1);
        dev_buffer[4 * idx + 2] += light.z / ((float)sa + 1);

        dev_samples[idx]++;
    }

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
    texturesSetup(scene, dev_scene);
    hdriSetup(scene, dev_scene);

    printf("%dMB of texture data copied\n", (textureMemory / (1024 * 1024)));

    int tx = THREADSIZE;
    int ty = THREADSIZE;

    dim3 blocks(camera->xRes / tx + 1, camera->yRes / ty + 1);
    dim3 threads(tx, ty);

    // Launch a kernel on the GPU with one thread for each element.
    setupKernel << <blocks, threads, 0, kernelStream >> > ();

    cudaStatus = cudaStreamSynchronize(kernelStream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

Error:

    return cudaStatus;
}

//TODO quitar variables globales pasando por parámetro el puntero.

void renderCuda(Scene* scene, int sampleTarget) {

    int tx = THREADSIZE;
    int ty = THREADSIZE;

    dim3 blocks(scene->camera.xRes / tx + 1, scene->camera.yRes / ty + 1);
    dim3 threads(tx, ty);

    for (int i = 0; i < sampleTarget; i++) {

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
        fprintf(stderr, "returned error code %d after launching addKernel!\n", cudaStatus);
    }

    cudaStatus = cudaMemcpyFromSymbolAsync(pathcountBuffer, dev_pathcount, size * sizeof(unsigned int), 0, cudaMemcpyDeviceToHost, bufferStream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "returned error code %d after launching addKernel!\n", cudaStatus);
    }

    return cudaStatus;
}

int getSamples() {

    int buff;

    cudaStreamCreate(&bufferStream);

    cudaError_t cudaStatus = cudaMemcpyFromSymbolAsync(&buff, dev_samples, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost, bufferStream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "returned error code %d after launching addKernel!\n", cudaStatus);
    }

    return buff;
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

    //hitdata.normal = Vector3(0, 1, 0);

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