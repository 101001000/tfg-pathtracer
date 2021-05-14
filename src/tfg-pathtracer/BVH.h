#ifndef BVH_H
#define BVH_H

#include <limits>
#include "Tri.h"
#include "Math.h"
#include "Ray.h"
#include <vector>
#include <algorithm>
#include <chrono>  

#define DEPTH 16
#define SAHBINS 24

struct BVHTri {

	Tri tri;
	int index;

	BVHTri(Tri _tri, int _index) {
		tri = _tri;
		index = _index;
	}

	BVHTri() {

	}

};

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

// Data structure which holds all the geometry data organized so it can be intersected fast with light rays.

class BVH {

public:

	int nodeIdx = 0;
	int allocatedTris = 0;
	int totalTris = 0;

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
			return false;

		// if tmin > tmax, ray doesn't intersect AABB
		if (tmin > tmax)
			return false;

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

			if (node.depth == (DEPTH - 1) && rOverlap)
				intersectNode(ray, rChild, nearestHit);

			if (node.depth == (DEPTH - 1) && lOverlap)
				intersectNode(ray, lChild, nearestHit);
			
			bool traverseL = (lOverlap && node.depth != (DEPTH - 1));
			bool traverseR = (rOverlap && node.depth != (DEPTH - 1));

			if (!traverseL && !traverseR) {
				node = *--stackPtr;

			} else {
				node = (traverseL) ? lChild : rChild;
				if (traverseL && traverseR)
					*stackPtr++ = rChild;
			}

		} while (node.valid);
	}

	__host__ __device__ void intersectNode(Ray ray, Node node, Hit& nearestHit) {

		for (int i = node.from; i < node.to; i++) {

			Hit hit;

			if (tris[triIndices[i]].hit(ray, hit, Vector3())) {

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

	void build(std::vector<Tri>* _fullTris) {

		std::vector<BVHTri>* tris = new std::vector<BVHTri>();

		for (int i = 0; i < _fullTris->size(); i++) {
			BVHTri bvhTri(_fullTris->at(i), i);
			tris->push_back(bvhTri);
		}			

		buildIt(tris);
	}
	
	void buildIt(std::vector<BVHTri>* _tris) {

		struct BuildNode {
			Node node;
			std::vector<BVHTri>* tris;

			BuildNode(Node _node, std::vector<BVHTri>* _tris) {
				node = _node;
				tris = _tris;
			}

			BuildNode() {

			}
		};

		BuildNode stack[64];

		int stackPointer = 0;

		Vector3 b1, b2, b3, b4;

		bounds(_tris, b1, b2);

		stack[stackPointer++] = BuildNode(Node(-1, b1, b2, 0, 0, 0), _tris);

		while (stackPointer > 0) {

			// POP
			BuildNode node = stack[--stackPointer];

			node.node.idx = nodeIdx++;

			if (node.node.depth == 0)
				totalTris = node.tris->size();

			if (node.node.depth == 7)
				printf("Allocated tris: %d / %d, %d%% \n", allocatedTris, totalTris, (100 * allocatedTris) / totalTris);

			// Nodo hoja
			if (node.node.depth == DEPTH) {

				node.node.from = triIdx;
				node.node.to = triIdx + node.tris->size();

				for (int i = 0; i < node.tris->size(); i++)
					triIndices[triIdx++] = node.tris->at(i).index;

				allocatedTris += node.tris->size();
			}
			else {

				std::vector<BVHTri>* trisLeft = new std::vector<BVHTri>();
				std::vector<BVHTri>* trisRight = new std::vector<BVHTri>();

				divideSAH(node.tris, trisLeft, trisRight);

				bounds(trisLeft, b1, b2);
				bounds(trisRight, b3, b4);

				BuildNode leftNode = BuildNode(Node(-1, b1, b2, 0, 0, node.node.depth + 1), trisLeft);
				BuildNode rightNode = BuildNode(Node(-1, b3, b4, 0, 0, node.node.depth + 1), trisRight);

				stack[stackPointer++] = rightNode;
				stack[stackPointer++] = leftNode;
			}

			node.tris->clear();

			delete node.tris;

			nodes[node.node.idx] = node.node;
		}
	}


	
	void buildAux(int depth, std::vector<BVHTri>* _tris) {

		if (depth == 0)
			totalTris = _tris->size();

		if(depth == 7)
			printf("Allocated tris: %d / %d, %d%% \n", allocatedTris, totalTris, (100 * allocatedTris) / totalTris);

		Vector3 b1;
		Vector3 b2;

		bounds(_tris, b1, b2);

		if (depth == DEPTH) {

			nodes[nodeIdx] = Node(nodeIdx, b1, b2, triIdx, triIdx + _tris->size(), depth);

			nodeIdx++;

			for (int i = 0; i < _tris->size(); i++)
				triIndices[triIdx++] = _tris->at(i).index;

			allocatedTris += _tris->size();
		}
		else {

			nodes[nodeIdx] = Node(nodeIdx, b1, b2, 0, 0, depth);

			nodeIdx++;

			std::vector<BVHTri>* trisLeft = new std::vector<BVHTri>();
			std::vector<BVHTri>* trisRight = new std::vector<BVHTri>();

			divideNaive(_tris, trisLeft, trisRight);

			buildAux(depth + 1, trisLeft);
			buildAux(depth + 1, trisRight);

			trisLeft->clear();
			trisRight->clear();

			delete trisLeft;
			delete trisRight;
		}
	}

	static void dividePlane(std::vector<BVHTri>* tris, std::vector<BVHTri>* trisLeft, std::vector<BVHTri>* trisRight) {

		Vector3 b1, b2;

		bounds(tris, b1, b2);

		// Buscamos que dimensión de la caja que envuelve a todos los triángulos es la más grande y partimos por la mitad de ahí

		float lx = b2.x - b1.x;
		float ly = b2.y - b1.y;
		float lz = b2.z - b1.z;

		if (lx > ly && lx > lz) {

			for (int i = 0; i < tris->size(); i++) {
				if (tris->at(i).tri.centroid().x > b1.x + lx / 2) {
					trisLeft->push_back(tris->at(i));
				}
				else {
					trisRight->push_back(tris->at(i));
				}
			}

		} else if (ly > lx && ly > lz) {
			for (int i = 0; i < tris->size(); i++) {
				if (tris->at(i).tri.centroid().y > b1.y + ly / 2) {
					trisLeft->push_back(tris->at(i));
				}
				else {
					trisRight->push_back(tris->at(i));
				}
			}
		}
		else {
			for (int i = 0; i < tris->size(); i++) {
				if (tris->at(i).tri.centroid().z > b1.z + lz / 2) {
					trisLeft->push_back(tris->at(i));
				}
				else {
					trisRight->push_back(tris->at(i));
				}
			}
		}
	}

	static void divideSAH(std::vector<BVHTri>* tris, std::vector<BVHTri>* trisLeft, std::vector<BVHTri>* trisRight) {

		if (tris->size() <= 0) {
			return;
		}

		Vector3 totalB1;
		Vector3 totalB2;

		int bestBin = 0;
		int bestAxis = 0;

		float bestHeuristic = 1000000000000;

		bounds(tris, totalB1, totalB2);

		for (int axis = 0; axis < 3; axis++) {

			Vector3 b1s[SAHBINS];
			Vector3 b2s[SAHBINS];

			int count[SAHBINS];

			for (int i = 0; i < SAHBINS; i++) {
				count[i] = 0;
				b1s[i] = Vector3();
				b2s[i] = Vector3();
			}

		

			for (int i = 0; i < tris->size(); i++) {

				// El bin en el que cae
				int bin;
				
				if (totalB1[axis] == totalB2[axis]) {
					bin = 0;
				}	else {
					float c = tris->at(i).tri.centroid()[axis];
					bin = map(c, totalB1[axis], totalB2[axis], 0, SAHBINS - 1);
				}
			
				count[bin]++;

				Vector3 b1, b2;

				bounds(tris->at(i).tri, b1, b2);

				boundsUnion(b1s[bin], b2s[bin], b1, b2, b1s[bin], b2s[bin]);
			}

	
			for (int i = 0; i < SAHBINS; i++) {

				int count1 = 0;
				int count2 = 0;

				Vector3 b1, b2, b3, b4;

				for (int j = 0; j < i; j++) {
					count1 += count[j];
					boundsUnion(b1, b2, b1s[j], b2s[j], b1, b2);
				}

				for (int k = i; k < SAHBINS; k++) {
					count2 += count[k];
					boundsUnion(b3, b4, b1s[k], b2s[k], b3, b4);
				}

				float heuristic = boundsArea(b1, b2) * (float)count1 + boundsArea(b3, b4) * (float)count2;

				if (heuristic < bestHeuristic) {
					bestHeuristic = heuristic;
					bestBin = i;
					bestAxis = axis;
				}
			}
		}
		
		for (int i = 0; i < tris->size(); i++) {

			// El bin en el que cae
			float c = tris->at(i).tri.centroid()[bestAxis];

			int bin = map(c, totalB1[bestAxis], totalB2[bestAxis], 0, SAHBINS - 1);

			//printf("Cae en %d\n", bin);

			if (bin < bestBin) {
				trisLeft->push_back(tris->at(i));
			}
			else {
				trisRight->push_back(tris->at(i));
			}
		}
	}

	static void divideNaive(std::vector<BVHTri>* tris, std::vector<BVHTri>* trisLeft, std::vector<BVHTri>* trisRight) {

		// Cogemos un triángulo al azar, buscamos el más lejano y los convertimos en centroides de la división

	
		int furtherIdx = 0;

		for (int i = 0; i < tris->size(); i++) {

			Vector3 c0 = tris->at(0).tri.centroid();
			Vector3 ci = tris->at(i).tri.centroid();
			Vector3 cf = tris->at(furtherIdx).tri.centroid();

			if ((c0 - ci).length() > (c0 - cf).length()){
				furtherIdx = i;
			}	
		}
	

		for (int i = 0; i < tris->size(); i++) {

			if ((tris->at(0).tri.centroid() - tris->at(i).tri.centroid()).length() < (tris->at(furtherIdx).tri.centroid() - tris->at(i).tri.centroid()).length()) {
				trisLeft->push_back(tris->at(i));
			}
			else {
				trisRight->push_back(tris->at(i));
			}
		}
	}

	static void boundsUnion(Vector3 b1, Vector3 b2, Vector3 b3, Vector3 b4, Vector3& b5, Vector3& b6) {

		if (boundsArea(b1, b2) <= 0 || boundsArea(b3, b4) <= 0) {

			if (boundsArea(b1, b2) <= 0) {
				b5 = b3;
				b6 = b4;
			}

			if (boundsArea(b3, b4) <= 0) {
				b5 = b1;
				b6 = b2;
			}
		}
		else {

			b5.x = minf(b1.x, minf(b2.x, minf(b3.x, b4.x)));
			b5.y = minf(b1.y, minf(b2.y, minf(b3.y, b4.y)));
			b5.z = minf(b1.z, minf(b2.z, minf(b3.z, b4.z)));

			b6.x = maxf(b1.x, maxf(b2.x, maxf(b3.x, b4.x)));
			b6.y = maxf(b1.y, maxf(b2.y, maxf(b3.y, b4.y)));
			b6.z = maxf(b1.z, maxf(b2.z, maxf(b3.z, b4.z)));
		}

	}

	static float boundsArea(Vector3 b1, Vector3 b2) {

		float x = b2.x - b1.x;
		float y = b2.y - b1.y;
		float z = b2.z - b1.z;

		return 2 * (x * y + x * z + y * z);

	}

	static void bounds(Tri tri, Vector3& b1, Vector3& b2) {

		b1.x = minf(tri.vertices[0].x, minf(tri.vertices[1].x, tri.vertices[2].x));
		b1.y = minf(tri.vertices[0].y, minf(tri.vertices[1].y, tri.vertices[2].y));
		b1.z = minf(tri.vertices[0].z, minf(tri.vertices[1].z, tri.vertices[2].z));

		b2.x = maxf(tri.vertices[0].x, maxf(tri.vertices[1].x, tri.vertices[2].x));
		b2.y = maxf(tri.vertices[0].y, maxf(tri.vertices[1].y, tri.vertices[2].y));
		b2.z = maxf(tri.vertices[0].z, maxf(tri.vertices[1].z, tri.vertices[2].z));
	}

	static void bounds(std::vector<BVHTri>* tris, Vector3& b1, Vector3& b2) {

	
		if (tris->size() <= 0) {
			//b1 = Vector3();
			//b2 = Vector3();
			return;
		}

		b1 = tris->at(0).tri.vertices[0];
		b2 = tris->at(0).tri.vertices[0];

		for (int i = 0; i < tris->size(); i++) {

			b1.x = minf(tris->at(i).tri.vertices[0].x, b1.x);
			b1.y = minf(tris->at(i).tri.vertices[0].y, b1.y);
			b1.z = minf(tris->at(i).tri.vertices[0].z, b1.z);

			b1.x = minf(tris->at(i).tri.vertices[1].x, b1.x);
			b1.y = minf(tris->at(i).tri.vertices[1].y, b1.y);
			b1.z = minf(tris->at(i).tri.vertices[1].z, b1.z);

			b1.x = minf(tris->at(i).tri.vertices[2].x, b1.x);
			b1.y = minf(tris->at(i).tri.vertices[2].y, b1.y);
			b1.z = minf(tris->at(i).tri.vertices[2].z, b1.z);

			b2.x = maxf(tris->at(i).tri.vertices[0].x, b2.x);
			b2.y = maxf(tris->at(i).tri.vertices[0].y, b2.y);
			b2.z = maxf(tris->at(i).tri.vertices[0].z, b2.z);

			b2.x = maxf(tris->at(i).tri.vertices[1].x, b2.x);
			b2.y = maxf(tris->at(i).tri.vertices[1].y, b2.y);
			b2.z = maxf(tris->at(i).tri.vertices[1].z, b2.z);

			b2.x = maxf(tris->at(i).tri.vertices[2].x, b2.x);
			b2.y = maxf(tris->at(i).tri.vertices[2].y, b2.y);
			b2.z = maxf(tris->at(i).tri.vertices[2].z, b2.z);
		}
	}

};


#endif

