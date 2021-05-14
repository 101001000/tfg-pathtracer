#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>

#include <stdio.h>

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

dpct::global_memory<float, 1> dev_buffer(1920 * 1080 * 4);

// How many samples per pixels has been calculated.
dpct::global_memory<unsigned int, 1> dev_samples(1920 * 1080);
dpct::global_memory<dev_Scene, 1> dev_scene_g;
/*
DPCT1032:1: Different generator is used, you may need to adjust the code.
*/
dpct::global_memory<curandState, 1> d_rand_state_g;

sycl::queue *kernelStream, bufferStream;

void createHitData(Material* material, HitData& hitdata, float u, float v, Vector3 N,
                   dev_Scene *dev_scene_g) {

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

void setupKernel(sycl::nd_item<3> item_ct1, float *dev_buffer,
                 unsigned int *dev_samples, dev_Scene *dev_scene_g,
                 curandState *d_rand_state_g) {

    int x = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
    int y = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range().get(1);

    int idx = (dev_scene_g->camera->xRes * y + x);

    if ((x >= dev_scene_g->camera->xRes) || (y >= dev_scene_g->camera->yRes)) return;

    dev_samples[idx] = 0;

    dev_buffer[4 * idx + 0] = 0;
    dev_buffer[4 * idx + 1] = 0;
    dev_buffer[4 * idx + 2] = 0;
    dev_buffer[4 * idx + 3] = 1;

    //Inicialización rápida de curand, se pierden propiedades matemáticas o algo así
    d_rand_state_g[idx] =
        oneapi::mkl::rng::device::philox4x32x10<1>(idx, {0, 0 * 8});

    if (x == 0 && y == 0) {

        int triSum = 0;

        for (int i = 0; i < dev_scene_g->meshObjectCount; i++) {
            dev_scene_g->meshObjects[i].tris += triSum;
            triSum += dev_scene_g->meshObjects[i].triCount;
        }
    }
}

Hit throwRay(Ray ray, dev_Scene* scene) {

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
Vector3 directLight(Ray ray, HitData hitdata, dev_Scene* scene, Vector3 point, float& pdf, float r1, Vector3& newDir) {

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
Vector3 hdriLight(Ray ray, dev_Scene* scene, Vector3 point, HitData hitdata, float r1, float r2, float r3, float& pdf) {

    if (!HDRIIS) {

        Vector3 newDir = UniformSampleSphere(r1, r2).normalized();

        float u, v;

        Texture::sphericalMapping(Vector3(), -1 * newDir, 1, u, v);

        Ray shadowRay(point + newDir * 0.001, newDir);

        Hit shadowHit = throwRay(shadowRay, scene);

        if (shadowHit.valid) return Vector3();

        Vector3 hdriValue = scene->hdri->texture.getValueBilinear(u, v);

        Vector3 brdfDisney = DisneyEval(ray, hitdata, newDir);

        return brdfDisney * sycl::fabs(Vector3::dot(newDir, hitdata.normal)) *
               hdriValue / (1.0 / (2.0 * PI));
    }
    else {

        Vector3 textCoordinate = scene->hdri->sample(r1);

        float nu = circle((float)textCoordinate.x / (float)scene->hdri->texture.width) + ((r2 - 0.5) / (float)scene->hdri->texture.width);
        float nv = circle((float)textCoordinate.y / (float)scene->hdri->texture.height)+ ((r3 - 0.5) / (float)scene->hdri->texture.height);

        Vector3 newDir = -1 * Texture::reverseSphericalMapping(nu, nv).normalized();

        Ray shadowRay(point + newDir * 0.001, newDir);

        Hit shadowHit = throwRay(shadowRay, scene);

        if (shadowHit.valid) return Vector3();

        Vector3 hdriValue = scene->hdri->texture.getValueBilinear(nu, nv);

        Vector3 brdfDisney = DisneyEval(ray, hitdata, newDir);

        pdf = scene->hdri->pdf(textCoordinate.x, textCoordinate.y);

        return brdfDisney * sycl::fabs(Vector3::dot(newDir, hitdata.normal)) *
               hdriValue / pdf;
    }
}

void neeRenderKernel(sycl::nd_item<3> item_ct1, float *dev_buffer,
                     unsigned int *dev_samples, dev_Scene *dev_scene_g,
                     curandState *d_rand_state_g){

    oneapi::mkl::rng::device::uniform<float> distr_ct1;
    dev_Scene *scene = (*dev_scene_g);

    int x = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
    int y = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range().get(1);

    if ((x >= scene->camera->xRes) || (y >= scene->camera->yRes)) return;

    // The index for the pixel. Maybe I could look for another indexing function which preserves better the spaciality
    int idx = (scene->camera->xRes * y + x);

    /*
    DPCT1032:2: Different generator is used, you may need to adjust the code.
    */
    oneapi::mkl::rng::device::philox4x32x10<1> local_rand_state =
        (*d_rand_state_g)[idx];

    // Relative coordinates for the point where the first ray will be launched
    float dx = scene->camera->position.x + ((float)x) / ((float)scene->camera->xRes) * scene->camera->sensorWidth * 0.001;
    float dy = scene->camera->position.y + ((float)y) / ((float)scene->camera->yRes) * scene->camera->sensorHeight * 0.001;

    // Absolute coordinates for the point where the first ray will be launched
    float odx = (-scene->camera->sensorWidth / 2) * 0.001 + dx;
    float ody = (-scene->camera->sensorHeight / 2) * 0.001 + dy;

    // Random part of the sampling offset so we get antialasing
    float rx =
        (1.0 / (float)scene->camera->xRes) *
        (oneapi::mkl::rng::device::generate(distr_ct1, local_rand_state) -
         0.5) *
        scene->camera->sensorWidth * 0.001;
    float ry =
        (1.0 / (float)scene->camera->yRes) *
        (oneapi::mkl::rng::device::generate(distr_ct1, local_rand_state) -
         0.5) *
        scene->camera->sensorHeight * 0.001;

    // The initial ray is created from the camera position to the point calculated before. No rotation is taken into account.
    Ray ray = Ray(scene->camera->position, Vector3(odx + rx, ody + ry, scene->camera->position.z + scene->camera->focalLength * 0.001) - scene->camera->position);

    float diameter = 0.001 * ((scene->camera->focalLength) / scene->camera->aperture);

    float l = (scene->camera->focusDistance + scene->camera->focalLength * 0.001);

    Vector3 focusPoint = ray.origin + ray.direction * (l/(ray.direction.z));

    float ix;
    float iy;

    uniformCircleSampling(
        oneapi::mkl::rng::device::generate(distr_ct1, local_rand_state),
        oneapi::mkl::rng::device::generate(distr_ct1, local_rand_state),
        oneapi::mkl::rng::device::generate(distr_ct1, local_rand_state), ix,
        iy);

    Vector3 or = scene->camera->position + diameter * Vector3(ix * 0.5, iy * 0.5, 0);

    ray = Ray(or, focusPoint - or);


    // Accumulated radiance
    Vector3 light = Vector3(0, 0, 0);

    // How much light is lost in the path
    Vector3 reduction = Vector3(1, 1, 1);

    // A ray can bounce a max of MAXBOUNCES. This could be changed with russian roulette path termination method, and that would make
    // the renderer unbiased
    for (int i = 0; i < MAXBOUNCES; i++) {

        oneapi::mkl::rng::device::uniform<float> distr_ct2;
        int materialID = 0;

        HitData hitdata;

        Hit nearestHit = throwRay(ray, scene);
        Vector3 cN = nearestHit.normal;

        // FIX BACKFACE NORMALS
        if (Vector3::dot(cN, ray.direction) > 0) cN *= -1;


        if (!nearestHit.valid) {
            float u, v;
            float w1 = 0.5;
            Texture::sphericalMapping(Vector3(), -1 * ray.direction, 1, u, v);
            light += w1 * scene->hdri->texture.getValueBilinear(u, v) * reduction;
            break;
        }

        if (nearestHit.type == 0) materialID = scene->spheres[nearestHit.objectID].materialID;
        if (nearestHit.type == 1) materialID = scene->meshObjects[nearestHit.objectID].materialID;

        Material* material = &scene->materials[materialID];

        createHitData(material, hitdata, nearestHit.u, nearestHit.v, cN, dev_scene_g);

        float r1 = oneapi::mkl::rng::device::generate(distr_ct2, local_rand_state);
        float r2 = oneapi::mkl::rng::device::generate(distr_ct2, local_rand_state);
        float r3 = oneapi::mkl::rng::device::generate(distr_ct2, local_rand_state);
        float r4 = oneapi::mkl::rng::device::generate(distr_ct2, local_rand_state);
        float r5 = oneapi::mkl::rng::device::generate(distr_ct2, local_rand_state);
        float r6 = oneapi::mkl::rng::device::generate(distr_ct2, local_rand_state);
        float r7 = oneapi::mkl::rng::device::generate(distr_ct2, local_rand_state);

        Vector3 brdfDir = DisneySample(ray, hitdata, r3, r4, r5);;
        Vector3 brdfDisney = DisneyEval(ray, hitdata, brdfDir);

        float brdfPdf = DisneyPdf(ray, hitdata, brdfDir);
        float hdriPdf;

        Vector3 directLight = hdriLight(ray, scene, nearestHit.position, hitdata, r1, r6, r7, hdriPdf);

        float w1 = hdriPdf / (hdriPdf + brdfPdf);
        float w2 = brdfPdf / (hdriPdf + brdfPdf);

        light += w1 * reduction * directLight;
         
        if (brdfPdf <= 0) break;

        reduction *=
            w2 * (brdfDisney * sycl::fabs(Vector3::dot(brdfDir, cN))) / brdfPdf;

        ray = Ray(nearestHit.position + brdfDir * 0.001, brdfDir);
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

    (*d_rand_state_g)[idx] = local_rand_state;
}


// TODO: BUSCAR FORMA DE SIMPLIFICAR EL SETUP
void pointLightsSetup(Scene *scene, dev_Scene *dev_scene) {
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();

    unsigned int pointLightCount = scene->pointLightCount();

    PointLight* dev_pointLights;

    dpct::get_default_queue()
        .memcpy(&dev_scene->pointLightCount, &pointLightCount,
                sizeof(unsigned int))
        .wait();

    dev_pointLights = sycl::malloc_device<PointLight>(
        pointLightCount, dpct::get_default_queue());
    dpct::get_default_queue()
        .memcpy(dev_pointLights, scene->getPointLights(),
                sizeof(PointLight) * pointLightCount)
        .wait();

    dpct::get_default_queue()
        .memcpy(&(dev_scene->pointLights), &(dev_pointLights),
                sizeof(PointLight *))
        .wait();
}
void materialsSetup(Scene *scene, dev_Scene *dev_scene) {
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();

    unsigned int materialCount = scene->materialCount();

    Material* dev_materials;

    dpct::get_default_queue()
        .memcpy(&dev_scene->materialCount, &materialCount, sizeof(unsigned int))
        .wait();

    dev_materials =
        sycl::malloc_device<Material>(materialCount, dpct::get_default_queue());

    dpct::get_default_queue()
        .memcpy(dev_materials, scene->getMaterials(),
                sizeof(Material) * materialCount)
        .wait();

    dpct::get_default_queue()
        .memcpy(&(dev_scene->materials), &(dev_materials), sizeof(Material *))
        .wait();
}
void spheresSetup(Scene *scene, dev_Scene *dev_scene) {
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();

    unsigned int sphereCount = scene->sphereCount();

    Sphere* spheres = scene->getSpheres();

    Sphere* dev_spheres;

    dpct::get_default_queue()
        .memcpy(&dev_scene->sphereCount, &sphereCount, sizeof(unsigned int))
        .wait();

    dev_spheres =
        sycl::malloc_device<Sphere>(sphereCount, dpct::get_default_queue());

    dpct::get_default_queue()
        .memcpy(dev_spheres, spheres, sizeof(Sphere) * sphereCount)
        .wait();

    dpct::get_default_queue()
        .memcpy(&(dev_scene->spheres), &(dev_spheres), sizeof(Sphere *))
        .wait();
}
void texturesSetup(Scene *scene, dev_Scene *dev_scene) {
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();

    unsigned int textureCount = scene->textureCount();

    Texture* textures = scene->getTextures();

    Texture* dev_textures;

    dpct::get_default_queue()
        .memcpy(&dev_scene->textureCount, &textureCount, sizeof(unsigned int))
        .wait();

    dev_textures =
        sycl::malloc_device<Texture>(textureCount, dpct::get_default_queue());

    dpct::get_default_queue()
        .memcpy(dev_textures, textures, sizeof(Texture) * textureCount)
        .wait();

    for (int i = 0; i < textureCount; i++) {

        float* textureData;

        textureData = (float *)sycl::malloc_device(
            sizeof(float) * textures[i].width * textures[i].height * 3,
            dpct::get_default_queue());

        dpct::get_default_queue()
            .memcpy(textureData, textures[i].data,
                    sizeof(float) * textures[i].width * textures[i].height * 3)
            .wait();
        dpct::get_default_queue()
            .memcpy(&(dev_textures[i].data), &textureData, sizeof(float *))
            .wait();

        printf("Texture %d copied, %dpx x %dpx\n", i, textures[i].width, textures[i].height);
    }

    dpct::get_default_queue()
        .memcpy(&(dev_scene->textures), &(dev_textures), sizeof(Texture *))
        .wait();
}
void hdriSetup(Scene *scene, dev_Scene *dev_scene) {
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();

    HDRI* hdri = &scene->hdri;

    HDRI* dev_hdri;

    float* dev_data; 
    float* dev_cdf;

    dev_hdri = sycl::malloc_device<HDRI>(1, dpct::get_default_queue());
    dev_data = (float *)sycl::malloc_device(
        sizeof(float) * hdri->texture.height * hdri->texture.width * 3,
        dpct::get_default_queue());
    dev_cdf = (float *)sycl::malloc_device(
        sizeof(float) * hdri->texture.height * hdri->texture.width,
        dpct::get_default_queue());

    dpct::get_default_queue().memcpy(dev_hdri, hdri, sizeof(HDRI)).wait();
    dpct::get_default_queue()
        .memcpy(dev_data, hdri->texture.data,
                sizeof(float) * hdri->texture.height * hdri->texture.width * 3)
        .wait();
    dpct::get_default_queue()
        .memcpy(dev_cdf, hdri->cdf,
                sizeof(float) * hdri->texture.height * hdri->texture.width)
        .wait();

    dpct::get_default_queue()
        .memcpy(&(dev_hdri->texture.data), &(dev_data), sizeof(float *))
        .wait();
    dpct::get_default_queue()
        .memcpy(&(dev_hdri->cdf), &(dev_cdf), sizeof(float *))
        .wait();
    dpct::get_default_queue()
        .memcpy(&(dev_scene->hdri), &(dev_hdri), sizeof(float *))
        .wait();
}

int renderSetup(Scene *scene) try {
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();

    printf("Initializing rendering... \n");

    kernelStream = dpct::get_current_device().create_queue();

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
    dev_scene = sycl::malloc_device<dev_Scene>(1, dpct::get_default_queue());

    int cudaStatus;

    /*
    DPCT1032:3: Different generator is used, you may need to adjust the code.
    */
    oneapi::mkl::rng::device::philox4x32x10<1> *d_rand_state;
    /*
    DPCT1032:4: Different generator is used, you may need to adjust the code.
    */
    d_rand_state =
        sycl::malloc_device<oneapi::mkl::rng::device::philox4x32x10<1>>(
            camera->xRes * camera->yRes, dpct::get_default_queue());

    dpct::get_default_queue()
        .memcpy(&dev_scene->meshObjectCount, &meshObjectCount,
                sizeof(unsigned int))
        .wait();
    dpct::get_default_queue()
        .memcpy(&dev_scene->triCount, &triCount, sizeof(unsigned int))
        .wait();

    std::cout << "Copying..." << std::endl;

    // Choose which GPU to run on, change this on a multi-GPU system.
    /*
    DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    cudaStatus = (dpct::dev_mgr::instance().select_device(0), 0);

    /*
    DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    cudaStatus =
        (dev_camera = sycl::malloc_device<Camera>(1, dpct::get_default_queue()),
         0);
    /*
    DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    cudaStatus = (dev_meshObjects = sycl::malloc_device<MeshObject>(
                      meshObjectCount, dpct::get_default_queue()),
                  0);

    /*
    DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    cudaStatus = (dev_tris = sycl::malloc_device<Tri>(
                      triCount, dpct::get_default_queue()),
                  0);
    /*
    DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    cudaStatus =
        (dev_bvh = sycl::malloc_device<BVH>(1, dpct::get_default_queue()), 0);
    /*
    DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dev_triIndices = sycl::malloc_device<int>(
                      triCount, dpct::get_default_queue()),
                  0);

    printf("Memory allocated... \n");

    /*
    DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dpct::get_default_queue()
                      .memcpy(dev_meshObjects, meshObjects,
                              sizeof(MeshObject) * meshObjectCount)
                      .wait(),
                  0);
    /*
    DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dpct::get_default_queue()
                      .memcpy(dev_camera, camera, sizeof(Camera))
                      .wait(),
                  0);
    /*
    DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dpct::get_default_queue()
                      .memcpy(dev_tris, tris, sizeof(Tri) * triCount)
                      .wait(),
                  0);
    /*
    DPCT1003:14: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus =
        (dpct::get_default_queue().memcpy(dev_bvh, bvh, sizeof(BVH)).wait(), 0);
    /*
    DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus =
        (dpct::get_default_queue()
             .memcpy(dev_triIndices, bvh->triIndices, sizeof(int) * triCount)
             .wait(),
         0);

    printf("Memory copied... \n");

    for (int i = 0; i < meshObjectCount; i++) {
        dpct::get_default_queue()
            .memcpy(&(dev_meshObjects[i].tris), &dev_tris, sizeof(Tri *))
            .wait();
    }

    //Pointer binding for dev_scene

    /*
    DPCT1003:16: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dpct::get_default_queue()
                      .memcpy(&(dev_scene->meshObjects), &(dev_meshObjects),
                              sizeof(MeshObject *))
                      .wait(),
                  0);
    /*
    DPCT1003:17: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus =
        (dpct::get_default_queue()
             .memcpy(&(dev_scene->camera), &(dev_camera), sizeof(Camera *))
             .wait(),
         0);
    /*
    DPCT1003:18: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dpct::get_default_queue()
                      .memcpy(&(dev_scene->tris), &(dev_tris), sizeof(Tri *))
                      .wait(),
                  0);
    /*
    DPCT1003:19: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dpct::get_default_queue()
                      .memcpy(&(dev_scene->bvh), &(dev_bvh), sizeof(BVH *))
                      .wait(),
                  0);

    /*
    DPCT1003:20: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dpct::get_default_queue()
                      .memcpy(&(dev_bvh->tris), &(dev_tris), sizeof(Tri *))
                      .wait(),
                  0);
    /*
    DPCT1003:21: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus =
        (dpct::get_default_queue()
             .memcpy(&(dev_bvh->triIndices), &(dev_triIndices), sizeof(int *))
             .wait(),
         0);

    /*
    DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus =
        (dpct::get_default_queue()
             .memcpy(dev_scene_g.get_ptr(), &dev_scene, sizeof(dev_Scene *))
             .wait(),
         0);
    /*
    DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1032:24: Different generator is used, you may need to adjust the code.
    */
    cudaStatus =
        (dpct::get_default_queue()
             .memcpy(d_rand_state_g.get_ptr(), &d_rand_state,
                     sizeof(oneapi::mkl::rng::device::philox4x32x10<1> *))
             .wait(),
         0);

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

    sycl::range<3> blocks(1, camera->yRes / ty + 1, camera->xRes / tx + 1);
    sycl::range<3> threads(1, ty, tx);

    // Launch a kernel on the GPU with one thread for each element.
    setupKernel << <blocks, threads, 0, kernelStream >> >();

    /*
    DPCT1003:25: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (kernelStream->wait(), 0);

Error:

    return cudaStatus;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void renderCuda(Scene *scene) try {

    int tx = THREADSIZE;
    int ty = THREADSIZE;

    sycl::range<3> blocks(1, scene->camera.yRes / ty + 1,
                          scene->camera.xRes / tx + 1);
    sycl::range<3> threads(1, ty, tx);

    while (1) {

        neeRenderKernel << <blocks, threads, 0, kernelStream >> > ();

        /*
        DPCT1003:26: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        int cudaStatus = (kernelStream->wait(), 0);
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int getBuffer(float *buffer, int size) try {

    bufferStream = dpct::get_current_device().create_queue();

    std::cout << "getting Buffer" << std::endl;

    /*
    DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    int cudaStatus =
        (bufferStream->memcpy(buffer, dev_buffer.get_ptr(*bufferStream),
                              size * sizeof(float)),
         0);

    return cudaStatus;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int getSamples() {

    int buff[5];

    cudaStreamCreate(&stream2);

    std::cout << "getting Buffer" << std::endl;

    int cudaStatus = cudaMemcpyFromSymbolAsync(
        buff, dev_samples, sizeof(int), 0, cudaMemcpyDeviceToHost, stream2);

    return buff[0];
}

void printHDRISampling(HDRI hdri, int samples) {

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

void printBRDFMaterial(Material material, int samples) {

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