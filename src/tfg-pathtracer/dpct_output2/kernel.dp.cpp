#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>

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
#include <cmath>

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

dpct::global_memory<float, 1> dev_buffer(1920 * 1080 * 4);

// How many samples per pixels has been calculated.
dpct::global_memory<unsigned int, 1> dev_samples(1920 * 1080);
dpct::global_memory<unsigned int, 1> dev_pathcount(1920 * 1080);
dpct::global_memory<dev_Scene *, 0> dev_scene_g;
/*
DPCT1032:2: A different random number generator is used. You may need to adjust
the code.
*/
dpct::global_memory<curandState *, 0> d_rand_state_g;

sycl::queue *kernelStream, bufferStream;

long textureMemory = 0;
long geometryMemory = 0;

void generateHitData(Material* material, HitData& hitdata, Hit hit,
                     dev_Scene **dev_scene_g) {

    Vector3 tangent, bitangent, normal;

    normal = hit.normal;
    tangent = hit.tangent;
    bitangent = hit.bitangent;

    if (material->albedoTextureID < 0) {
        hitdata.albedo = material->albedo;
    }
    else {
        hitdata.albedo = dev_scene_g->textures[material->albedoTextureID].getValueFromUVFiltered(hit.tu, hit.tv);
    }

    if (material->emissionTextureID < 0) {
        hitdata.emission = material->emission;
    }
    else {
        hitdata.emission = dev_scene_g->textures[material->emissionTextureID].getValueFromUVFiltered(hit.tu, hit.tv);
    }

    if (material->roughnessTextureID < 0) {
        hitdata.roughness = material->roughness;
    }
    else {
        hitdata.roughness = dev_scene_g->textures[material->roughnessTextureID].getValueFromUVFiltered(hit.tu, hit.tv).x;
    }

    if (material->metallicTextureID < 0) {
        hitdata.metallic = material->metallic;
    }
    else {
        hitdata.metallic = dev_scene_g->textures[material->metallicTextureID].getValueFromUVFiltered(hit.tu, hit.tv).x;
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
    hitdata.roughness = sycl::pow<double>(hitdata.roughness, 2.2f);
    hitdata.metallic = sycl::pow<double>(hitdata.metallic, 2.2f);

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

void setupKernel(sycl::nd_item<3> item_ct1, float *dev_buffer,
                 unsigned int *dev_samples, unsigned int *dev_pathcount,
                 dev_Scene **dev_scene_g, curandState **d_rand_state_g) {

    int x = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
    int y = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range().get(1);

    int idx = (dev_scene_g->camera->xRes * y + x);

    if ((x >= dev_scene_g->camera->xRes) || (y >= dev_scene_g->camera->yRes)) return;

    dev_samples[idx] = 0;
    dev_pathcount[idx] = 0;

    memset(&dev_buffer[4 * idx], 0, sizeof(float) * 3); // RGB channels
    dev_buffer[4 * idx + 3] = 1; // Alpha channel

    d_rand_state_g[idx] = oneapi::mkl::rng::device::philox4x32x10<1>(
        0, {0, static_cast<std::uint64_t>(idx * 8)});

    // Just one thread
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


Vector3 pointLight(Ray ray, HitData hitdata, dev_Scene* scene, Vector3 point, float& pdf, float r1) {

    PointLight light = scene->pointLights[(int)(scene->pointLightCount * r1)];

    pdf = scene->pointLightCount / (2.0 * PI);

    Vector3 newDir = (light.position - point).normalized();

    float dist = (light.position - point).length();

    Vector3 pointLightValue = (light.radiance / (dist * dist));

    Vector3 brdfDisney = DisneyEval(ray, hitdata, newDir);

    return pointLightValue * brdfDisney *
           sycl::fabs(Vector3::dot(newDir, hitdata.normal)) / pdf;
}

// Sampling HDRI
// The main idea is to get a random point of the HDRI, weighted by their importance and then get the direction from the center to that point as if that pixel
// would be in a sphere of infinite radius.
Vector3 hdriLight(Ray ray, dev_Scene* scene, Vector3 point, HitData hitdata, float r1, float r2, float r3, float& pdf) {

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

        return brdfDisney * sycl::fabs(Vector3::dot(newDir, hitdata.normal)) *
               hdriValue / (1.0 / (2.0 * PI));
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

        return brdfDisney * sycl::fabs(Vector3::dot(newDir, hitdata.normal)) *
               hdriValue / pdf;
    }
}

void calculateCameraRay(int x, int y, Camera& camera, Ray& ray, float r1, float r2, float r3, float r4, float r5) {

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

void shade(dev_Scene& scene, Ray& ray, HitData& hitdata, Hit& nearestHit, Vector3& newDir, float r1, float r2, float r3, Vector3& hitLight, Vector3& reduction) {

    Vector3 brdfDisney = DisneyEval(ray, hitdata, newDir);

    float brdfPdf = DisneyPdf(ray, hitdata, newDir);

    float hdriPdf;
    float pointPdf;

    Vector3 hdriLightCalc = hdriLight(ray, &scene, nearestHit.position, hitdata, r1, r2, r3, hdriPdf);
    Vector3 pointLightCalc = pointLight(ray, hitdata, &scene, nearestHit.position, r1, pointPdf);

    float w1 = hdriPdf / (hdriPdf + brdfPdf);
    float w2 = brdfPdf / (hdriPdf + brdfPdf);

    if (brdfPdf <= 0) return;

    brdfPdf = 1/(2*PI);

   // hitLight = reduction * hdriLightCalc;

    reduction *=
        (brdfDisney * sycl::fabs(Vector3::dot(newDir, hitdata.normal))) /
        brdfPdf;
}

void calculateBounce(Ray& incomingRay, HitData& hitdata, Vector3& bouncedDir, float r1, float r2, float r3) {

    //bouncedDir = DisneySample(incomingRay, hitdata, r1, r2, r3);
    //bouncedDir = CosineSampleHemisphere(r1, r2).normalized();
    bouncedDir = uniformSampleSphere(r1, r2).normalized();

    if (Vector3::dot(hitdata.normal, bouncedDir) < 0) bouncedDir *= -1;
}

void renderingKernel(sycl::nd_item<3> item_ct1, float *dev_buffer,
                     unsigned int *dev_samples, unsigned int *dev_pathcount,
                     dev_Scene **dev_scene_g, curandState **d_rand_state_g) {

    oneapi::mkl::rng::device::uniform<float> distr_ct1;
    dev_Scene *scene = (*dev_scene_g);

    int x = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
    int y = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range().get(1);

    if ((x >= scene->camera->xRes) || (y >= scene->camera->yRes)) return;

    int idx = (scene->camera->xRes * y + x);

    /*
    DPCT1032:3: A different random number generator is used. You may need to
    adjust the code.
    */
    oneapi::mkl::rng::device::philox4x32x10<1> local_rand_state =
        (*d_rand_state_g)[idx];

    unsigned int sa = dev_samples[idx];

    Ray ray;

    calculateCameraRay(
        x, y, *scene->camera, ray,
        oneapi::mkl::rng::device::generate(distr_ct1, local_rand_state),
        oneapi::mkl::rng::device::generate(distr_ct1, local_rand_state),
        oneapi::mkl::rng::device::generate(distr_ct1, local_rand_state),
        oneapi::mkl::rng::device::generate(distr_ct1, local_rand_state),
        oneapi::mkl::rng::device::generate(distr_ct1, local_rand_state));

    // Accumulated radiance
    Vector3 light = Vector3::Zero();

    // How much light is lost in the path
    Vector3 reduction = Vector3::One();

    // A ray can bounce a max of MAXBOUNCES. This could be changed with russian roulette path termination method, and that would make
    // the renderer unbiased

    int i = 0;

    for (i = 0; i < MAXBOUNCES; i++) {

        oneapi::mkl::rng::device::uniform<float> distr_ct2;
        Vector3 hitLight;
        HitData hitdata;
        Vector3 bouncedDir;

        int materialID = 0;

        Hit nearestHit = throwRay(ray, scene);

        if (!nearestHit.valid) {
            float u, v;
            Texture::sphericalMapping(Vector3(), -1 * ray.direction, 1, u, v);
            light += scene->hdri->texture.getValueFromUVFiltered(u, v) * reduction;
            break;
        }

        //if (Vector3::dot(nearestHit.normal, ray.direction) > 0) nearestHit.normal *= -1;

        materialID = scene->meshObjects[nearestHit.objectID].materialID;

        Material* material = &scene->materials[materialID];

        generateHitData(material, hitdata, nearestHit, dev_scene_g);

        calculateBounce(
            ray, hitdata, bouncedDir,
            oneapi::mkl::rng::device::generate(distr_ct2, local_rand_state),
            oneapi::mkl::rng::device::generate(distr_ct2, local_rand_state),
            oneapi::mkl::rng::device::generate(distr_ct2, local_rand_state));

        shade(*scene, ray, hitdata, nearestHit, bouncedDir,
              oneapi::mkl::rng::device::generate(distr_ct2, local_rand_state),
              oneapi::mkl::rng::device::generate(distr_ct2, local_rand_state),
              oneapi::mkl::rng::device::generate(distr_ct2, local_rand_state),
              hitLight, reduction);

        float hdriPdf;

        //Vector3 hLight = hdriLight(ray, scene, nearestHit.position, hitdata, curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), hdriPdf);;

        light += hitLight;

        ray = Ray(nearestHit.position + bouncedDir * 0.001, bouncedDir);
    }

    dev_pathcount[idx] += i;

    //light = clamp(light, 0, 10);

    if (!sycl::isnan(light.x) && !sycl::isnan(light.y) && !sycl::isnan(light.z)) {

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
    textureMemory += sizeof(Texture) * textureCount;

    dpct::get_default_queue()
        .memcpy(dev_textures, textures, sizeof(Texture) * textureCount)
        .wait();

    for (int i = 0; i < textureCount; i++) {

        float* textureData;

        textureData = (float *)sycl::malloc_device(
            sizeof(float) * textures[i].width * textures[i].height * 3,
            dpct::get_default_queue());
        textureMemory += sizeof(float) * textures[i].width * textures[i].height * 3;

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

    textureMemory += sizeof(HDRI) + sizeof(float) * hdri->texture.height * hdri->texture.width * 4;

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
    DPCT1032:5: A different random number generator is used. You may need to
    adjust the code.
    */
    oneapi::mkl::rng::device::philox4x32x10<1> *d_rand_state;
    /*
    DPCT1032:6: A different random number generator is used. You may need to
    adjust the code.
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

    printf("Copying data to device\n");

    // Choose which GPU to run on, change this on a multi-GPU system.
    /*
    DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    cudaStatus = (dpct::dev_mgr::instance().select_device(0), 0);

    /*
    DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    cudaStatus =
        (dev_camera = sycl::malloc_device<Camera>(1, dpct::get_default_queue()),
         0);
    /*
    DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    cudaStatus = (dev_meshObjects = sycl::malloc_device<MeshObject>(
                      meshObjectCount, dpct::get_default_queue()),
                  0);

    /*
    DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dev_tris = sycl::malloc_device<Tri>(
                      triCount, dpct::get_default_queue()),
                  0);
    /*
    DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus =
        (dev_bvh = sycl::malloc_device<BVH>(1, dpct::get_default_queue()), 0);
    /*
    DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dev_triIndices = sycl::malloc_device<int>(
                      triCount, dpct::get_default_queue()),
                  0);

    geometryMemory += sizeof(MeshObject) * meshObjectCount + sizeof(Tri) * triCount + sizeof(BVH) + sizeof(int) * triCount + sizeof(BVH);

    /*
    DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dpct::get_default_queue()
                      .memcpy(dev_meshObjects, meshObjects,
                              sizeof(MeshObject) * meshObjectCount)
                      .wait(),
                  0);
    /*
    DPCT1003:14: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dpct::get_default_queue()
                      .memcpy(dev_camera, camera, sizeof(Camera))
                      .wait(),
                  0);
    /*
    DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dpct::get_default_queue()
                      .memcpy(dev_tris, tris, sizeof(Tri) * triCount)
                      .wait(),
                  0);
    /*
    DPCT1003:16: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus =
        (dpct::get_default_queue().memcpy(dev_bvh, bvh, sizeof(BVH)).wait(), 0);
    /*
    DPCT1003:17: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus =
        (dpct::get_default_queue()
             .memcpy(dev_triIndices, bvh->triIndices, sizeof(int) * triCount)
             .wait(),
         0);

    for (int i = 0; i < meshObjectCount; i++) {
        dpct::get_default_queue()
            .memcpy(&(dev_meshObjects[i].tris), &dev_tris, sizeof(Tri *))
            .wait();
    }

    //Pointer binding for dev_scene

    /*
    DPCT1003:18: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dpct::get_default_queue()
                      .memcpy(&(dev_scene->meshObjects), &(dev_meshObjects),
                              sizeof(MeshObject *))
                      .wait(),
                  0);
    /*
    DPCT1003:19: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus =
        (dpct::get_default_queue()
             .memcpy(&(dev_scene->camera), &(dev_camera), sizeof(Camera *))
             .wait(),
         0);
    /*
    DPCT1003:20: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dpct::get_default_queue()
                      .memcpy(&(dev_scene->tris), &(dev_tris), sizeof(Tri *))
                      .wait(),
                  0);
    /*
    DPCT1003:21: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dpct::get_default_queue()
                      .memcpy(&(dev_scene->bvh), &(dev_bvh), sizeof(BVH *))
                      .wait(),
                  0);

    /*
    DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (dpct::get_default_queue()
                      .memcpy(&(dev_bvh->tris), &(dev_tris), sizeof(Tri *))
                      .wait(),
                  0);
    /*
    DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus =
        (dpct::get_default_queue()
             .memcpy(&(dev_bvh->triIndices), &(dev_triIndices), sizeof(int *))
             .wait(),
         0);

    /*
    DPCT1003:24: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus =
        (dpct::get_default_queue()
             .memcpy(dev_scene_g.get_ptr(), &dev_scene, sizeof(dev_Scene *))
             .wait(),
         0);
    /*
    DPCT1003:25: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1032:26: A different random number generator is used. You may need to
    adjust the code.
    */
    cudaStatus =
        (dpct::get_default_queue()
             .memcpy(d_rand_state_g.get_ptr(), &d_rand_state,
                     sizeof(oneapi::mkl::rng::device::philox4x32x10<1> *))
             .wait(),
         0);

    printf("%ldMB of geometry data copied\n", (geometryMemory / (1024L * 1024L)));

    pointLightsSetup(scene, dev_scene);
    materialsSetup(scene, dev_scene);
    texturesSetup(scene, dev_scene);
    hdriSetup(scene, dev_scene);

    printf("%dMB of texture data copied\n", (textureMemory / (1024 * 1024)));

    int tx = THREADSIZE;
    int ty = THREADSIZE;

    sycl::range<3> blocks(1, camera->yRes / ty + 1, camera->xRes / tx + 1);
    sycl::range<3> threads(1, ty, tx);

    // Launch a kernel on the GPU with one thread for each element.
    /*
    DPCT1049:4: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    kernelStream->submit([&](sycl::handler &cgh) {
        dev_buffer.init(*kernelStream);
        dev_samples.init(*kernelStream);
        dev_pathcount.init(*kernelStream);
        dev_scene_g.init(*kernelStream);
        d_rand_state_g.init(*kernelStream);

        auto dev_buffer_ptr_ct1 = dev_buffer.get_ptr();
        auto dev_samples_ptr_ct1 = dev_samples.get_ptr();
        auto dev_pathcount_ptr_ct1 = dev_pathcount.get_ptr();
        auto dev_scene_g_ptr_ct1 = dev_scene_g.get_ptr();
        auto d_rand_state_g_ptr_ct1 = d_rand_state_g.get_ptr();

        cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             setupKernel(
                                 item_ct1, dev_buffer_ptr_ct1,
                                 dev_samples_ptr_ct1, dev_pathcount_ptr_ct1,
                                 dev_scene_g_ptr_ct1, d_rand_state_g_ptr_ct1);
                         });
    });

    /*
    DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (kernelStream->wait(), 0);

    return cudaStatus;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//TODO quitar variables globales pasando por parámetro el puntero.

void renderCuda(Scene *scene, int sampleTarget) try {

    int tx = THREADSIZE;
    int ty = THREADSIZE;

    sycl::range<3> numBlocks(1, scene->camera.yRes / ty + 1,
                             scene->camera.xRes / tx + 1);
    sycl::range<3> threadsPerBlock(1, ty, tx);

    for (int i = 0; i < sampleTarget; i++) {

        /*
        DPCT1049:28: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        kernelStream->submit([&](sycl::handler &cgh) {
            dev_buffer.init(*kernelStream);
            dev_samples.init(*kernelStream);
            dev_pathcount.init(*kernelStream);
            dev_scene_g.init(*kernelStream);
            d_rand_state_g.init(*kernelStream);

            auto dev_buffer_ptr_ct1 = dev_buffer.get_ptr();
            auto dev_samples_ptr_ct1 = dev_samples.get_ptr();
            auto dev_pathcount_ptr_ct1 = dev_pathcount.get_ptr();
            auto dev_scene_g_ptr_ct1 = dev_scene_g.get_ptr();
            auto d_rand_state_g_ptr_ct1 = d_rand_state_g.get_ptr();

            cgh.parallel_for(
                sycl::nd_range<3>(numBlocks * threadsPerBlock, threadsPerBlock),
                [=](sycl::nd_item<3> item_ct1) {
                    renderingKernel(item_ct1, dev_buffer_ptr_ct1,
                                    dev_samples_ptr_ct1, dev_pathcount_ptr_ct1,
                                    dev_scene_g_ptr_ct1,
                                    d_rand_state_g_ptr_ct1);
                });
        });

        /*
        DPCT1003:29: Migrated API does not return error code. (*, 0) is
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

int getBuffer(float *pixelBuffer, int *pathcountBuffer, int size) try {

    bufferStream = dpct::get_current_device().create_queue();

    /*
    DPCT1003:30: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    int cudaStatus =
        (bufferStream->memcpy(pixelBuffer, dev_buffer.get_ptr(*bufferStream),
                              size * sizeof(float) * 4),
         0);

    /*
    DPCT1003:31: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    cudaStatus = (bufferStream->memcpy(pathcountBuffer,
                                       dev_pathcount.get_ptr(*bufferStream),
                                       size * sizeof(unsigned int)),
                  0);

    return cudaStatus;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int getSamples() try {

    int buff;

    bufferStream = dpct::get_current_device().create_queue();

    /*
    DPCT1003:32: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    int cudaStatus =
        (bufferStream->memcpy(&buff, dev_samples.get_ptr(*bufferStream),
                              sizeof(unsigned int)),
         0);

    return buff;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
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