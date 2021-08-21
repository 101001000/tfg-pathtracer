#ifndef SCENEOBJECT_H
#define SCENEOBJECT_H

#include "Vector.hpp"

static int objectIDCount = 0;

class SceneObject {

public:

	Vector3 position, rotation, scale;

	int objectID = 0;;

	SceneObject() {

		objectID = objectIDCount++;

		//TODO reemplazar por presets
		position = Vector3();
		rotation = Vector3();
		scale = Vector3(1, 1, 1);
	}


};

#endif