#ifndef  OBJLOADER_H
#define OBJLOADER_H

#include "mikktspaceCallback.h"
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>

#include "Vector.hpp"
#include "mikktspace.h"
#include "Tri.hpp"
#include "MeshObject.hpp"

struct UnloadedMaterial {
	Material mat;
	std::map<std::string, std::string> maps;
};

class ObjLoader {

public:

	std::vector<Vector3> vertices;
	std::vector<Vector3> textureCoord;
	std::vector<Vector3> normals;

	static std::string getSecondWord(std::string str) {
		std::string::size_type in = str.find_first_of(" ");
		return str.substr(in + 1, str.length());
	}

	UnloadedMaterial parseMtl(std::ifstream& stream, std::string name) {

		UnloadedMaterial umtl;

		umtl.mat.name = name;

		std::string line;
		while (std::getline(stream, line) && !(line.find("newmtl") != std::string::npos)) {

			//Ns TODO this is an exponent for specular
			//Ka TODO --> Ambient color
			//Kd
			if (line[0] == 'K' && line[1] == 'd')
				umtl.mat.albedo = Vector3(line.substr(2));
			//Ks TODO find a way to make this compatible
			if (line[0] == 'K' && line[1] == 's')
				umtl.mat.specular = Vector3(line.substr(2)).x;
			//Ke
			if (line[0] == 'K' && line[1] == 'e')
				umtl.mat.emission = Vector3(line.substr(2));
			//Ni
			if (line[0] == 'N' && line[1] == 'i')
				umtl.mat.eta = std::stof(getSecondWord(line));
			//d
			if (line[0] == 'd')
				umtl.mat.opacity = std::stof(getSecondWord(line)); // Implicit type conversion, check if it's appropiate
			
			std::vector<std::string> mapNames = { "map_Kd", "map_Ns", "map_Bump", "refl" };

			for (int i = 0; i < mapNames.size(); i++) {
				if (line.find(mapNames[i]) != std::string::npos)
					umtl.maps[mapNames[i]] = getSecondWord(line);
			}
		}

		return umtl;
	}

	MeshObject parseObj(std::ifstream &stream) {

		std::streampos pos = stream.tellg();
		std::string line;
		std::vector<Tri>* tris = new std::vector<Tri>();

		MeshObject mo;

		while (std::getline(stream, line) && line[0] != 'o') {

			if (line.find("usemtl") != std::string::npos)
				mo.matName = getSecondWord(line);

			if (line[0] == 'v' && line[1] == ' ')
				vertices.push_back(Vector3(line.substr(2)) * Vector3(1, 1, -1));

			if (line[0] == 'v' && line[1] == 't')
				textureCoord.push_back(Vector3(line.substr(2)));

			if (line[0] == 'v' && line[1] == 'n')
				normals.push_back(Vector3(line.substr(2)) * Vector3(1, 1, -1));

			if (line[0] == 'f') {

				char f[3][1000];

				f[0][0] = '\0';
				f[1][0] = '\0';
				f[2][0] = '\0';

				Tri tri;

				int idx = -1;

				for (char& c : line) {

					if (isdigit(c) || c == '/') {

						if (idx == -1) idx = 0;
						int len = strlen(f[idx]);

						f[idx][len] = c;
						f[idx][len + 1] = '\0';
					}
					else if (idx > -1) {
						idx++;
					}
				}

				for (int i = 0; i < 3; i++) {

					char v[3][100];

					v[0][0] = '\0';
					v[1][0] = '\0';
					v[2][0] = '\0';

					int _idx = 0;

					for (char& c : f[i]) {
						if (c == '\0') break;

						if (isdigit(c)) {
							int len = strlen(v[_idx]);

							v[_idx][len] = c;
							v[_idx][len + 1] = '\0';
						}
						else {
							_idx++;
						}
					}


					if (strlen(v[0]) > 0)
						tri.vertices[i] = vertices.at(std::stoi(&(v[0])[0]) - 1);

					if (strlen(v[1]) > 0)
						tri.uv[i] = textureCoord.at(std::stoi(&(v[1])[0]) - 1);

					if (strlen(v[2]) > 0)
						tri.normals[i] = normals.at(std::stoi(&(v[2])[0]) - 1).normalized();

				}
				tris->push_back(tri);
			}
		}

		printf("Obj loaded with %d tris, %d vertices and %d normals\n", tris->size(), vertices.size(), normals.size());

		mo.tris = tris->data();
		mo.triCount = tris->size();

		CalcTangents calcTang = CalcTangents();
		calcTang.calc(&mo);
	
		return mo;
	}

	std::vector<MeshObject> loadObjs(std::string path) {

		std::ifstream input(path.c_str());
		std::string line;
		std::vector<MeshObject> meshObjects;

		while (std::getline(input, line)) {

			if (line[0] == 'o') {
				std::streampos pos = input.tellg();
				meshObjects.push_back(parseObj(input));
				input.seekg(pos);
			}
				
		}
		return meshObjects;
	}

	std::vector<UnloadedMaterial> loadMtls(std::string path) {

		std::ifstream input(path.c_str());
		std::string line;
		std::vector<UnloadedMaterial> mtls;

		while (std::getline(input, line)) {
			if (line.find("newmtl") != std::string::npos) {
				std::streampos pos = input.tellg();
				mtls.push_back(parseMtl(input, getSecondWord(line)));
				input.seekg(pos);
			}
		}
		return mtls;
	}
};





#endif
