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

//TODO HACER ESTO CON MEMORIA DINÁMICA

__device__ float dev_buffer[1920 * 1080 * 4]; 
__device__ unsigned int dev_samples[1920 * 1080];
__device__ dev_Scene* dev_scene_g;
__device__ curandState* d_rand_state_g;

cudaStream_t stream1, stream2;

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

__device__ Vector3 hdriLight(Ray ray, dev_Scene* scene, Vector3 point, HitData hitdata, float r1, float r2) {
           
    Vector3 light;

    Vector3 textCoordinate = scene->hdri->sample(r1, r2);

    float nu = circle(1 - (textCoordinate.x / (float)scene->hdri->texture.width) - scene->hdri->texture.xOffset);
    float nv = (textCoordinate.y / (float)scene->hdri->texture.height) - scene->hdri->texture.yOffset;

    Vector3 newDir = Texture::reverseSphericalMapping(nu, nv).normalized();

    //if (Vector3::dot(hitdata.normal, newDir) < 0) return Vector3();

    Ray shadowRay(point + newDir * 0.01, newDir);

    Hit shadowHit = throwRay(shadowRay, scene);

    //if (shadowHit.valid) return Vector3();

    Vector3 hdriValue = scene->hdri->texture.getValue(textCoordinate.x, textCoordinate.y);

    float hdriPdf = scene->hdri->pdf(textCoordinate.x, textCoordinate.y);

    //printf("Valor %f, PDF, %f\n", hdriValue.length(), hdriPdf);

    Vector3 brdfDisney = DisneyEval(ray, hitdata, newDir);

    float pdfDisney = DisneyPdf(ray, hitdata, newDir);

    float misWeight = powerHeuristic(hdriPdf, pdfDisney);

    if (hdriPdf <= 0) return Vector3();

    light += misWeight * brdfDisney * abs(Vector3::dot(newDir, hitdata.normal)) * hdriValue / hdriPdf;

    return light;

    
    /*

    Vector3 light;

    Vector3 textCoordinate = scene->hdri->sample(r1, r2);

    float nu = textCoordinate.x / (float)scene->hdri->texture.width;
    float nv = textCoordinate.y / (float)scene->hdri->texture.height;

    Vector3 newDir = Texture::reverseSphericalMapping(nu, nv).normalized();

    newDir = UniformSampleSphere(r1, r2);

    if (Vector3::dot(hitdata.normal, newDir) < 0) newDir *= -1;

    Ray shadowRay(point + newDir * 0.001, newDir);

    Hit shadowHit = throwRay(shadowRay, scene);

    //if (shadowHit.valid) return Vector3();

    Vector3 hdriValue = scene->hdri->texture.getValue(textCoordinate.x, textCoordinate.y);

    float u, v;

    Texture::sphericalMapping(Vector3(), -1*newDir, 1, u, v);

    hdriValue = scene->hdri->texture.getValueBilinear(circle(1-u), v);

    float hdriPdf = scene->hdri->pdf(textCoordinate.x, textCoordinate.y);

    //hdriPdf = abs(Vector3::dot(newDir, hitdata.normal)) / (PI);

    Vector3 brdfDisney = DisneyEval(ray, hitdata, newDir);

    float pdfDisney = DisneyPdf(ray, hitdata, newDir);

    float misWeight = 1;// powerHeuristic(hdriPdf, pdfDisney);

    //hdriPdf = (scene->hdri->texture.width * scene->hdri->texture.height);

    //printf("Valor %f, PDF, %f\n", hdriValue.length(), hdriPdf);

    light += misWeight * brdfDisney * abs(Vector3::dot(newDir, hitdata.normal)) * hdriValue / (1.0);

    return light;*/
}

__global__ void neeRenderKernel(){

    dev_Scene* scene = dev_scene_g;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= scene->camera->xRes) || (y >= scene->camera->yRes)) return;

    int idx = (scene->camera->xRes * y + x);

    curandState local_rand_state = d_rand_state_g[idx];

    for (int s = 0; s < 1; s++) {

        float dx = scene->camera->position.x + ((float)x) / ((float)scene->camera->xRes) * scene->camera->sensorWidth * 0.001;
        float dy = scene->camera->position.y + ((float)y) / ((float)scene->camera->yRes) * scene->camera->sensorHeight * 0.001;
        
        float odx = (-scene->camera->sensorWidth / 2) * 0.001 + dx;
        float ody = (-scene->camera->sensorHeight / 2) * 0.001 + dy;

        float rx = (1 / (float)scene->camera->xRes) * (curand_uniform(&local_rand_state) - 0.5) * scene->camera->sensorWidth * 0.001;
        float ry = (1 / (float)scene->camera->yRes) * (curand_uniform(&local_rand_state) - 0.5) * scene->camera->sensorHeight * 0.001;

        Ray ray = Ray(scene->camera->position, Vector3(odx + rx, ody + ry, scene->camera->position.z + scene->camera->focalLength * 0.001) - scene->camera->position);

        Vector3 light = Vector3(0,0,0);
        Vector3 reduction = Vector3(1, 1, 1);

        float oldPdf = 1;

        for (int i = 0; i < MAXBOUNCES; i++) {

            int materialID = 0;
            HitData hitdata;

            Hit nearestHit = throwRay(ray, scene);
            Vector3 cN = nearestHit.normal;

            // FIX BACKFACE NORMALS
            if (Vector3::dot(cN, ray.direction) > 0) cN *= -1;

            // SAMPLE ENVIRONMENT
            
            if (!nearestHit.valid) {

                float misWeight = 1;

                float u, v;

                Texture::sphericalMapping(Vector3(), -1 * ray.direction, 1, u, v);

                Vector3 hdriValue = scene->hdri->texture.getValueBilinear(circle(1 - u), v);

                if (i > 0) {
                   
                    float hdriPdf = scene->hdri->pdf(u * scene->hdri->texture.width, v * scene->hdri->texture.height);

                    misWeight = powerHeuristic(oldPdf, hdriPdf);
                }               

                light += misWeight * hdriValue * reduction;

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

            Vector3 newDir;

            light += hdriLight(ray, scene, nearestHit.position, hitdata, r4, r5);

            newDir = DisneySample(ray, hitdata, r1, r2, r3);

            Vector3 brdfDisney = DisneyEval(ray, hitdata, newDir);

            float pdfDisney = DisneyPdf(ray, hitdata, newDir);

            oldPdf = pdfDisney;

            reduction *= (brdfDisney * abs(Vector3::dot(newDir, cN))) / pdfDisney;

            ray = Ray(nearestHit.position + newDir * 0.001, newDir);          
        } 

        light = clamp(light, 0, 3);

        unsigned int sa = dev_samples[idx];

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

    cudaMemcpy(dev_textures, textures, sizeof(Texture) * textureCount, cudaMemcpyHostToDevice);

    for (int i = 0; i < textureCount; i++) {

        float* textureData;

        cudaMalloc((void**)&textureData, sizeof(float) * textures[i].width * textures[i].height * 3);

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

    cudaMemcpy(dev_hdri, hdri, sizeof(HDRI), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data, hdri->texture.data, sizeof(float) * hdri->texture.height * hdri->texture.width * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cdf, hdri->cdf, sizeof(float) * hdri->texture.height * hdri->texture.width, cudaMemcpyHostToDevice);


    cudaMemcpy(&(dev_hdri->texture.data), &(dev_data), sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dev_hdri->cdf), &(dev_cdf), sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dev_scene->hdri), &(dev_hdri), sizeof(float*), cudaMemcpyHostToDevice);
}

cudaError_t renderSetup(Scene* scene) {

    printf("Initializing rendering... \n");

    cudaStreamCreate(&stream1);

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


    std::cout << "Copying..." << std::endl;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);

    cudaStatus = cudaMalloc((void**)&dev_camera, sizeof(Camera));
    cudaStatus = cudaMalloc((void**)&dev_meshObjects, sizeof(MeshObject) * meshObjectCount);

    cudaStatus = cudaMalloc((void**)&dev_tris, sizeof(Tri) * triCount);
    cudaStatus = cudaMalloc((void**)&dev_bvh, sizeof(BVH));
    cudaStatus = cudaMalloc((void**)&dev_triIndices, sizeof(int) * triCount);

    printf("Memory allocated... \n");

    cudaStatus = cudaMemcpy(dev_meshObjects, meshObjects, sizeof(MeshObject) * meshObjectCount, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_tris, tris, sizeof(Tri) * triCount, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_bvh, bvh, sizeof(BVH), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_triIndices, bvh->triIndices, sizeof(int) * triCount, cudaMemcpyHostToDevice);

    printf("Memory copied... \n");

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

    printf("pointers binded... \n");

    pointLightsSetup(scene, dev_scene);
    materialsSetup(scene, dev_scene);
    spheresSetup(scene, dev_scene);
    texturesSetup(scene, dev_scene);
    hdriSetup(scene, dev_scene);

    std::cout << "Copied" << std::endl;
    std::cout << "Running..." << std::endl;

    int tx = THREADSIZE;
    int ty = THREADSIZE;

    dim3 blocks(camera->xRes / tx + 1, camera->yRes / ty + 1);
    dim3 threads(tx, ty);

    // Launch a kernel on the GPU with one thread for each element.
    setupKernel << <blocks, threads, 0, stream1 >> >();

    cudaStatus = cudaStreamSynchronize(stream1);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

Error:

    return cudaStatus;
}

void renderCuda(Scene* scene) {

    int tx = THREADSIZE;
    int ty = THREADSIZE;

    dim3 blocks(scene->camera.xRes / tx + 1, scene->camera.yRes / ty + 1);
    dim3 threads(tx, ty);

    while (1) {

        neeRenderKernel << <blocks, threads, 0, stream1 >> > ();

        cudaError_t cudaStatus = cudaStreamSynchronize(stream1);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        }
    }

}

cudaError_t getBuffer(float* buffer, int size) {

    cudaStreamCreate(&stream2);

    std::cout << "getting Buffer" << std::endl;

    cudaError_t cudaStatus = cudaMemcpyFromSymbolAsync(buffer, dev_buffer, size * sizeof(float), 0, cudaMemcpyDeviceToHost, stream2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "LOL returned error code %d after launching addKernel!\n", cudaStatus);
    }

    return cudaStatus;
}

int getSamples() {

    int buff[5];

    cudaStreamCreate(&stream2);

    std::cout << "getting Buffer" << std::endl;

    cudaError_t cudaStatus = cudaMemcpyFromSymbolAsync(buff, dev_samples, sizeof(int), 0, cudaMemcpyDeviceToHost, stream2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "LOL returned error code %d after launching addKernel!\n", cudaStatus);
    }

    return buff[0];
}

