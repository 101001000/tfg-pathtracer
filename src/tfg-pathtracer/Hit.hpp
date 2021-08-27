#ifndef HIT_H
#define HIT_H

class Hit {

public:
	Vector3 position, normal, tangent, bitangent;
	bool valid = false;
	float t;
	unsigned int objectID;
	unsigned int type;
	float u, v;
};

#endif