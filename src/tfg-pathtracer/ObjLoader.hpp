#ifndef  OBJLOADER_H
#define OBJLOADER_H

#include <vector>
#include "Vector.hpp"
#include <sstream>
#include <iostream>
#include <fstream>

#include "Tri.hpp"
#include "MeshObject.hpp"

class ObjLoader {

public:

	static MeshObject loadObj(std::string path) {

		std::ifstream input(path.c_str());

		std::vector<Vector3> vertices;
		std::vector<Vector3> textureCoord;
		std::vector<Vector3> normals;
		std::vector<Tri>* tris = new std::vector<Tri>();

		std::string line;

		while (std::getline(input, line)) {

			if (line[0] == 'v' && line[1] == ' ') {
				vertices.push_back(Vector3(line.substr(2)) * Vector3(1, 1, -1));
			}

			if (line[0] == 'v' && line[1] == 't') {
				textureCoord.push_back(Vector3(line.substr(2)));
			}

			if (line[0] == 'v' && line[1] == 'n') {
				normals.push_back(Vector3(line.substr(2)) * Vector3(1, 1, -1));
			}

			if (line[0] == 'f') {

				char f[3][100];

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
					else if(idx > -1){

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

		MeshObject mo;

		mo.tris = tris->data();
		mo.triCount = tris->size();

		return mo;
	}
};





#endif
