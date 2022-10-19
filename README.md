# Eleven Renderer

Eleven Renderer is a small basic production oriented open source rendering engine coded in CUDA and C++. It has been made with academic and research purpose so feel free to contribute!

More info about how it works, in my Bachelor's degree (Spanish)
https://eprints.ucm.es/id/eprint/68327/1/CALLE%20MONTILLA%2063016_ENRIQUE_DE_LA_CALLE_MONTILLA_Evaluacion_y_aceleracion_del_algoritmo_Path_tracing_en_arquitecturas_heterogeneas_1006096_134067599.pdf

The last release of Eleven Renderer is https://github.com/101001000/tfg-pathtracer/releases/latest

The usage of the program is:

>eleven.exe <scene_path> <#samples> <output.bmp>

The repo includes couple scenes. For rendering f.e "ClockCC0", move the scene folder to the same path that the executable and type:

>eleven.exe "ClockCC0" 1000 "output.bmp"

![execfinalimage](https://user-images.githubusercontent.com/7725287/134220552-6e574522-64cd-4dda-a6ad-e21e46811f39.png)

You can compile it with Visual Studio 2019, use the MAKEFILE or try one of the releases

The main features implemented are the following ones:
* Texture maps (.bmp)
* HDRI Environment (.hdr)
* Defocus
* Disney BRDF
* Point lights
* Multiple importance sampling (NEE, BRDF, Environment, Pointlights)
* .obj file support (only tri faces)

I did some analysis with Nvidia Nsight Compute. This is the roofline for the ClockCC0 scene.
![](https://i.imgur.com/W0DPJhc.png)



Please keep in mind this project is in a very early phase.


LICENSING DISCLAIMER:

In this repo is included SFML library, which has https://opensource.org/licenses/Zlib zlib/png license, and Mikktspace, which also has a custom license. At the moment I'm working on sorting out files to make the repo fully GPL3, but at the moment it's not fully compilant.

HDRLoader and RSJp-cpp are also not my work, but they are licensed as GPL3.



