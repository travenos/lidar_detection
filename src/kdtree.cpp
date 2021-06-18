#include "kdtree.h"
#include <cassert>
#include <cmath>
#include <numeric>

// Structure to represent node of kd tree
struct Node
{
    std::vector<float> point;
    int id;
    Node* left;
    Node* right;

    Node(const std::vector<float>& arr, int setId)
        :	point{arr}, id{setId}, left{nullptr}, right{nullptr}
    {}

    ~Node()
    {
        delete left;
        delete right;
    }
private:
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
};

static void Insert(const std::vector<float>& point, int id, size_t k, Node*& node)
{
    if (node == nullptr)
    {
        node = new Node{point, id};
        return;
    }

    assert(point.size() == node->point.size());

    const size_t nextK = (k + 1) % point.size();
    if (point.at(k) < node->point.at(k))
    {
        ::Insert(point, id, nextK, node->left);
    }
    else
    {
        ::Insert(point, id, nextK, node->right);
    }
}

static bool IsFittingTolerance(const std::vector<float>& target, const std::vector<float>& otherNode, float distanceTol)
{
    assert(target.size() == otherNode.size());
    const size_t N{target.size()};

    bool isNear{false};
    std::vector<float> diffs(N);
    for (size_t i{0}; i < N; ++i)
    {
        diffs[i] = std::fabs(target[i] - otherNode[i]);
        isNear |= (diffs[i] <= distanceTol);
    }
    if (isNear)
    {
        const float squareSum = std::accumulate(diffs.begin(), diffs.end(), 0, [](float a, float b) {return a + b * b;});
        const float squareTol = distanceTol * distanceTol;
        isNear = (squareSum <= squareTol);
    }
    return isNear;
}

static void Search(const std::vector<float>& target, float distanceTol, size_t k, const Node* node, std::vector<int>& ids)
{
    if (node == nullptr)
    {
        return;
    }

    if (::IsFittingTolerance(target, node->point, distanceTol))
    {
        ids.push_back(node->id);
    }

    const size_t nextK = (k + 1) % target.size();
    if (target[k] - distanceTol < node->point[k])
    {
        ::Search(target, distanceTol, nextK, node->left, ids);
    }
    if (target[k] + distanceTol > node->point[k])
    {
        ::Search(target, distanceTol, nextK, node->right, ids);
    }
}

KdTree::~KdTree()
{
    delete root;
}

void KdTree::Insert(const std::vector<float>& point, int id)
{
    ::Insert(point, id, 0, root);
}

// return a list of point ids in the tree that are within distance of target
std::vector<int> KdTree::Search(const std::vector<float>& target, float distanceTol) const
{
    std::vector<int> ids;
    ::Search(target, distanceTol, 0, root, ids);
    return ids;
}



