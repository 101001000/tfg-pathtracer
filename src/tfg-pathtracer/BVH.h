#ifndef BVH_H
#define BVH_H

#include "Tri.h"
#include "Math.h"
#include "Ray.h"
#include <vector>
#include <algorithm>

#define DEPTH 16

struct Node {

	Vector3 b1, b2;
	int from, to, idx, depth;
	bool valid;

	__host__ __device__ Node() {
		valid = false;
	}

	__host__ __device__ Node(int _idx, Vector3 _b1, Vector3 _b2, int _from, int _to, int _depth) {
		idx = _idx;
		b1 = _b1;
		b2 = _b2;
		from = _from;
		to = _to;
		valid = true;
		depth = _depth;
	}
};

class BVH {

public:

	int nodeIdx = 0;

	Tri* tris;

	Node nodes[2 << (DEPTH + 2)];

	int triIdx = 0;
	int* triIndices;


	BVH() {

	}


	__host__ __device__ bool intersect(Ray ray, Vector3 b1, Vector3 b2) {

		//printf("Testing intersection with %f, %f, %f - %f, %f, %f\n", b1.x, b1.y, b1.z, b2.x, b2.y, b2.z);

		Vector3 dirfrac;

		// r.dir is unit direction vector of ray
		dirfrac.x = 1.0f / ray.direction.x;
		dirfrac.y = 1.0f / ray.direction.y;
		dirfrac.z = 1.0f / ray.direction.z;

		// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
		// r.org is origin of ray
		float t1 = (b1.x - ray.origin.x) * dirfrac.x;
		float t2 = (b2.x - ray.origin.x) * dirfrac.x;
		float t3 = (b1.y - ray.origin.y) * dirfrac.y;
		float t4 = (b2.y - ray.origin.y) * dirfrac.y;
		float t5 = (b1.z - ray.origin.z) * dirfrac.z;
		float t6 = (b2.z - ray.origin.z) * dirfrac.z;

		float tmin = maxf(maxf(minf(t1, t2), minf(t3, t4)), minf(t5, t6));
		float tmax = minf(minf(maxf(t1, t2), maxf(t3, t4)), maxf(t5, t6));

		// if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
		if (tmax < 0)
		{
			return false;
		}

		// if tmin > tmax, ray doesn't intersect AABB
		if (tmin > tmax)
		{
			return false;
		}

		return true;

	}

	__host__ __device__ void transverse(Ray ray, Hit& nearestHit) {

		Node stack[64];

		Node* stackPtr = stack;

		(*stackPtr++).valid = false;

		Node node = nodes[0];

		do {

			Node lChild = leftChild(node.idx, node.depth);
			Node rChild = rightChild(node.idx, node.depth);

			bool lOverlap = intersect(ray, lChild.b1, lChild.b2);
			bool rOverlap = intersect(ray, rChild.b1, rChild.b2);

			if (node.depth == (DEPTH - 1) && rOverlap) {
				//printf("INTERSECT RIGHT LEAF");
				intersectNode(ray, rChild, nearestHit);
			}

			if (node.depth == (DEPTH - 1) && lOverlap) {
				//printf("INTERSECT LEFT LEAF");
				intersectNode(ray, lChild, nearestHit);
			}
			
			// Query overlaps an internal node => traverse.
			bool traverseL = (lOverlap && node.depth != (DEPTH - 1));
			bool traverseR = (rOverlap && node.depth != (DEPTH - 1));

			if (!traverseL && !traverseR) {
				node = *--stackPtr; // pop

			} else {

				node = (traverseL) ? lChild : rChild;
				if (traverseL && traverseR)
					*stackPtr++ = rChild; // push
			}

		} while (node.valid);
	}

	__host__ __device__ void intersectNode(Ray ray, Node node, Hit& nearestHit) {

		for (int i = node.from; i < node.to; i++) {

			Hit hit;

			if (tris[triIndices[i]].hit(ray, hit, Vector3())) {

				//printf("HITVALID i %d with index: %d \n", i, triIndices[i]);

				if (!nearestHit.valid) {
					nearestHit = hit;
				}
				else if ((hit.position - ray.origin).length() < (nearestHit.position - ray.origin).length()) {
					nearestHit = hit;
				}
			}
		}
	}

	__host__ __device__ Node leftChild(int idx, int depth) {
		if (depth == DEPTH) return Node();
		return nodes[idx + 1];
	}

	__host__ __device__ Node rightChild(int idx, int depth) {
		if (depth == DEPTH) return Node();

		return nodes[idx + (2 << (DEPTH - depth - 1))];
	}
	
	void build(int depth, std::vector<Tri>* _tris, std::vector<Tri>* _fullTris) {

		Vector3 b1;
		Vector3 b2;

		bounds(_tris, b1, b2);

		//printf("BUILD DEPTH %d STEP 0 with BB ", depth);

		if (depth == DEPTH) {

			//printf("NODO %d desde %d hasta %d \n", nodeIdx, triIdx, triIdx + _tris->size());
			nodes[nodeIdx] = Node(nodeIdx, b1, b2, triIdx, triIdx + _tris->size(), depth);

			nodeIdx++;

			for (int i = 0; i < _tris->size(); i++) {

				int dis = std::distance(_fullTris->begin(), std::find(_fullTris->begin(), _fullTris->end(), _tris->at(i)));

				triIndices[triIdx++] = dis;
			}

			return;
		}
		else {

			nodes[nodeIdx] = Node(nodeIdx, b1, b2, 0, 0, depth);
			nodeIdx++;

			std::vector<Tri>* trisLeft = new std::vector<Tri>();
			std::vector<Tri>* trisRight = new std::vector<Tri>();

			divideNaive(_tris, trisLeft, trisRight);

			build(depth + 1, trisLeft, _fullTris);
			build(depth + 1, trisRight, _fullTris);

			trisLeft->clear();
			trisRight->clear();

			delete trisLeft;
			delete trisRight;
		}
	}


	static void dividePlane(std::vector<Tri>* tris, std::vector<Tri>* trisLeft, std::vector<Tri>* trisRight) {

		Vector3 b1, b2;

		bounds(tris, b1, b2);

		// Buscamos que dimensión de la caja que envuelve a todos los triángulos es la más grance y partimos por la mitad de ahí

		float lx = b2.x - b1.x;
		float ly = b2.y - b1.y;
		float lz = b2.z - b1.z;

		if (lx > ly && lx > lz) {

			for (int i = 0; i < tris->size(); i++) {
				if (tris->at(i).centroid().x > b1.x + lx / 2) {
					trisLeft->push_back(tris->at(i));
				}
				else {
					trisRight->push_back(tris->at(i));
				}
			}

		} else if (ly > lx && ly > lz) {
			for (int i = 0; i < tris->size(); i++) {
				if (tris->at(i).centroid().y > b1.y + ly / 2) {
					trisLeft->push_back(tris->at(i));
				}
				else {
					trisRight->push_back(tris->at(i));
				}
			}
		}
		else {
			for (int i = 0; i < tris->size(); i++) {
				if (tris->at(i).centroid().z > b1.z + lz / 2) {
					trisLeft->push_back(tris->at(i));
				}
				else {
					trisRight->push_back(tris->at(i));
				}
			}
		}
	}

	static void divideNaive(std::vector<Tri>* tris, std::vector<Tri>* trisLeft, std::vector<Tri>* trisRight) {

		// Cogemos un triángulo al azar, buscamos el más lejano y los convertimos en centroides de la división

		int furtherIdx = 0;

		for (int i = 0; i < tris->size(); i++) {

			Vector3 c0 = tris->at(0).centroid();
			Vector3 ci = tris->at(i).centroid();
			Vector3 cf = tris->at(furtherIdx).centroid();

			if ((c0 - ci).length() > (c0 - cf).length()){
				furtherIdx = i;
			}	
		}

		for (int i = 0; i < tris->size(); i++) {

			if ((tris->at(0).centroid() - tris->at(i).centroid()).length() < (tris->at(furtherIdx).centroid() - tris->at(i).centroid()).length()) {
				trisLeft->push_back(tris->at(i));
			}
			else {
				trisRight->push_back(tris->at(i));
			}
		}
	}

	static void bounds(std::vector<Tri>* tris, Vector3& b1, Vector3& b2) {

		if (tris->size() <= 0) return;

		b1 = tris->at(0).vertices[0];
		b2 = tris->at(0).vertices[0];

		for (int i = 0; i < tris->size(); i++) {

			b1.x = minf(tris->at(i).vertices[0].x, b1.x);
			b1.y = minf(tris->at(i).vertices[0].y, b1.y);
			b1.z = minf(tris->at(i).vertices[0].z, b1.z);

			b1.x = minf(tris->at(i).vertices[1].x, b1.x);
			b1.y = minf(tris->at(i).vertices[1].y, b1.y);
			b1.z = minf(tris->at(i).vertices[1].z, b1.z);

			b1.x = minf(tris->at(i).vertices[2].x, b1.x);
			b1.y = minf(tris->at(i).vertices[2].y, b1.y);
			b1.z = minf(tris->at(i).vertices[2].z, b1.z);

			b2.x = maxf(tris->at(i).vertices[0].x, b2.x);
			b2.y = maxf(tris->at(i).vertices[0].y, b2.y);
			b2.z = maxf(tris->at(i).vertices[0].z, b2.z);

			b2.x = maxf(tris->at(i).vertices[1].x, b2.x);
			b2.y = maxf(tris->at(i).vertices[1].y, b2.y);
			b2.z = maxf(tris->at(i).vertices[1].z, b2.z);

			b2.x = maxf(tris->at(i).vertices[2].x, b2.x);
			b2.y = maxf(tris->at(i).vertices[2].y, b2.y);
			b2.z = maxf(tris->at(i).vertices[2].z, b2.z);
		}
	}

};


#endif

