#ifndef KDTREE_H_
#define KDTREE_H_

//#include "../../render/render.h" //TODO!!!
#include <numeric>
#include <vector>

struct Node;

struct KdTree
{
    Node* root{nullptr};

    KdTree() = default;
    ~KdTree();
    void Insert(const std::vector<float>& point, int id);
	// return a list of point ids in the tree that are within distance of target
    std::vector<int> Search(const std::vector<float>& target, float distanceTol) const;
	
private:
    KdTree(const KdTree&) = delete;
    KdTree& operator=(const KdTree&) = delete;

};

#endif //KDTREE_H_
