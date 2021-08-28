#ifndef  RENDERABLEOBJECT_H
#define RENDERABLEOBJECT_H

// @todo Remove this class and fix dependencies

#include "SceneObject.hpp"


class RenderableObject : public SceneObject {

public:
	int materialID = 0;

	RenderableObject() {};

};

#endif
