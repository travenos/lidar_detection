#ifndef PROCESSPOINTCLOUDS_HPP_
#define PROCESSPOINTCLOUDS_HPP_

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>

#include <cassert>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <functional>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "render/box.h"
#include "kdtree.h"

template<typename PointT>
using PointCloudPtrTemplate = typename pcl::PointCloud<PointT>::Ptr;

template<typename PointT>
struct ProcessPointClouds
{
    using PointCloudPtr = PointCloudPtrTemplate<PointT>;

    static void NumPoints(const PointCloudPtr& cloud);

    static PointCloudPtr FilterCloud(const PointCloudPtr& cloud, float filterRes,
                                     const Eigen::Vector4f& minPoint, const Eigen::Vector4f& maxPoint,
                                     const Eigen::Vector4f& roofMin, const Eigen::Vector4f& roofMax);

    static std::pair<PointCloudPtr, PointCloudPtr> SeparateClouds(const pcl::PointIndices::Ptr& inliers, const PointCloudPtr& cloud);

    static std::pair<PointCloudPtr, PointCloudPtr> SegmentPlane(const PointCloudPtr& cloud, int maxIterations, float distanceThreshold);

    static void FillKdTree(KdTree* tree, const PointCloudPtr& cloud);

    static std::vector<PointCloudPtr> Clustering(const PointCloudPtr& cloud, float clusterTolerance, int minSize, int maxSize);

    static Box BoundingBox(const PointCloudPtr& cluster);

    static BoxQ BoundingBoxQ(const typename pcl::PointCloud<PointT>::Ptr& cluster);

    static void SavePcd(const PointCloudPtr& cloud, const std::string& file);

    static PointCloudPtr LoadPcd(const std::string& file);

    static std::vector<boost::filesystem::path> StreamPcd(const std::string& dataPath);
  
};


template<typename PointT>
void ProcessPointClouds<PointT>::NumPoints(const PointCloudPtr& cloud)
{
    std::cout << cloud->points.size() << std::endl;
}


template<typename PointT>
PointCloudPtrTemplate<PointT> ProcessPointClouds<PointT>::FilterCloud(const PointCloudPtr& cloud, float filterRes,
                                                                      const Eigen::Vector4f& minPoint, const Eigen::Vector4f& maxPoint,
                                                                      const Eigen::Vector4f& roofMin, const Eigen::Vector4f& roofMax)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    PointCloudPtr cloudFiltered{boost::make_shared<pcl::PointCloud<PointT>>()};
    // Create the filtering object
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize (filterRes, filterRes, filterRes);
    sor.filter (*cloudFiltered);

    PointCloudPtr cloudCropped{boost::make_shared<pcl::PointCloud<PointT>>()};
    pcl::CropBox<PointT> cropBox;
    cropBox.setMin(minPoint);
    cropBox.setMax(maxPoint);
    cropBox.setInputCloud(cloudFiltered);
    cropBox.filter(*cloudCropped);

    // Filter the roof
    pcl::PointIndices filteredIndices;
    cropBox.setMin(roofMin);
    cropBox.setMax(roofMax);
    cropBox.setInputCloud(cloudCropped);
    cropBox.setNegative(true);
    cropBox.filter(*cloudCropped);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloudCropped;

}

// Implementation of RANSAC
template<typename PointT>
std::pair<PointCloudPtrTemplate<PointT>, PointCloudPtrTemplate<PointT>> ProcessPointClouds<PointT>::SegmentPlane(const PointCloudPtr& cloud, int maxIterations, float distanceTol)
{
    // Helper structures
    struct Plane
    {
        Plane() = default;
        Plane (const pcl::PointCloud<PointT>& cloud, const std::unordered_set<int>& indices)
        {
            assert(indices.size() == 3);

            auto indIter = indices.begin();
            const float x1 = cloud.points[*indIter].x;
            const float y1 = cloud.points[*indIter].y;
            const float z1 = cloud.points[*indIter].z;
            ++indIter;
            const float x2 = cloud.points[*indIter].x;
            const float y2 = cloud.points[*indIter].y;
            const float z2 = cloud.points[*indIter].z;
            ++indIter;
            const float x3 = cloud.points[*indIter].x;
            const float y3 = cloud.points[*indIter].y;
            const float z3 = cloud.points[*indIter].z;

            A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
            B = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
            C = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
            D = - (A * x1 + B * y1 + C * z1);

            den = std::sqrt(A * A + B * B + C * C);
        }

        float getDistanceToPoint(const PointT& point) const
        {
            const float d = std::fabs(A * point.x + B * point.y + C * point.z + D) / den;
            return d;
        }

    private:
        float A{}, B{}, C{}, D{}, den{};
    };

    struct IterationResult
    {
        std::unordered_set<int> inliers{};
        Plane plane{};
    };

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    const int N{static_cast<int>(cloud->size())};
    assert(N > 2);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> randGen{0, N - 1};

    IterationResult bestResult;

    for (int i{0}; i < maxIterations; ++i)
    {
        IterationResult currentResult;

        while (currentResult.inliers.size() < 3)
        {
            currentResult.inliers.insert(randGen(rd));
        }

        currentResult.plane = Plane{*cloud, currentResult.inliers};

        for (int j{0}; j < N; ++j)
        {
            const auto& point = cloud->points[static_cast<std::size_t>(j)];
            const float pointDist = currentResult.plane.getDistanceToPoint(point);
            if (pointDist <= distanceTol)
            {
                currentResult.inliers.insert(j);
            }
        }

        if (currentResult.inliers.size() > bestResult.inliers.size())
        {
            bestResult = std::move(currentResult);
        }
    }

    pcl::PointIndices::Ptr inliers{boost::make_shared<pcl::PointIndices>()};
    inliers->indices.resize(bestResult.inliers.size());
    std::copy(bestResult.inliers.begin(), bestResult.inliers.end(), inliers->indices.begin());

    auto segResult = SeparateClouds(inliers, cloud);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "handmade implementation of plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    return segResult;
}


template<typename PointT>
std::pair<PointCloudPtrTemplate<PointT>, PointCloudPtrTemplate<PointT>> ProcessPointClouds<PointT>::SeparateClouds(const pcl::PointIndices::Ptr& inliers, const PointCloudPtr& cloud)
{
    PointCloudPtr obstCloud{boost::make_shared<pcl::PointCloud<PointT>>()};
    PointCloudPtr planeCloud{boost::make_shared<pcl::PointCloud<PointT>>()};
    // Extract the inliners

    // Create the filtering object
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*planeCloud);

    std::cout << "PointCloud representing the planar component: " << planeCloud->width * planeCloud->height << " data points." << std::endl;

    extract.setNegative(true);
    extract.filter(*obstCloud);

    std::pair<PointCloudPtr, PointCloudPtr> segResult{std::move(obstCloud), std::move(planeCloud)};
    return segResult;
}

template<typename PointT>
void ProcessPointClouds<PointT>::FillKdTree(KdTree* tree, const PointCloudPtr& cloud)
{
    struct Point
    {
        std::vector<float> coord;
        int id;
    };

    constexpr size_t DIM{3};
    assert(tree != nullptr);
    assert(!cloud->points.empty());
    const size_t N = cloud->points.size();
    const size_t maxLeafs = static_cast<size_t>(std::log2(N) + 1);

    std::vector<Point> idPoints(N);
    int currentId{};
    std::transform(cloud->points.begin(), cloud->points.end(), idPoints.begin(),
                   [&currentId](const PointT& point)
    {
        return Point{{point.x, point.y, point.z}, currentId++};
    });

    std::vector<std::vector<const Point*>> pointPointers(DIM + 1);
    for (auto& pointPointersDim : pointPointers)
    {
        pointPointersDim.resize(N);
    }

    using Iterator_t = typename std::vector<const Point*>::iterator;
    using Difference_t = typename std::vector<const Point*>::difference_type;
    std::vector<std::vector<std::pair<Iterator_t, Iterator_t>>> workingParts(DIM + 1);
    for (size_t i{0}; i < pointPointers.size() - 1; ++i)
    {
        std::transform(idPoints.begin(), idPoints.end(), pointPointers[i].begin(),
                       [](const Point& point){ return &point; });
        std::sort(pointPointers[i].begin(), pointPointers[i].end(),
                  [i](const Point* point1, const Point* point2) {return (point1->point)[i] < (point2->point)[i];});
        workingParts[i].reserve(maxLeafs);
        workingParts[i].emplace_back(pointPointers[i].begin(), pointPointers[i].end());
    }
    workingParts.back().reserve(maxLeafs);

    std::vector<const Point*> lastElements;
    lastElements.reserve(maxLeafs);

    size_t currentDim{DIM - 1};
    size_t nextDim{0};
    while (true)
    {
        bool noPointsAdded{false};
        if (!lastElements.empty())
        {
            noPointsAdded = true;
            const auto lastDim = currentDim;
            for (size_t checkingDim{0}; checkingDim < DIM; ++checkingDim)
            {
                if (checkingDim == lastDim)
                {
                    continue;
                }
                auto& tempPointPointers = pointPointers.back();
                auto& tempWorkingParts = workingParts.back();
                tempWorkingParts.clear();
                for (size_t i{0}; i < lastElements.size(); ++i)
                {
                    const auto& lastWorkingPart = workingParts[lastDim][i];
                    const Difference_t lastPartSizeMinus1 = lastWorkingPart.second - lastWorkingPart.first - 1;
                    auto firstBegin = !tempWorkingParts.empty() ? tempWorkingParts.back().second : tempPointPointers.begin();
                    auto firstEnd = firstBegin + std::max(static_cast<Difference_t>(0), lastPartSizeMinus1 / 2);
                    auto secondBegin = firstEnd;
                    auto secondEnd = secondBegin + std::max(static_cast<Difference_t>(0), lastPartSizeMinus1 - lastPartSizeMinus1 / 2);
                    tempWorkingParts.emplace_back(firstBegin, firstEnd);
                    tempWorkingParts.emplace_back(secondBegin, secondEnd);

                    auto currentIterFirst = firstBegin;
                    auto currentIterSecond = secondBegin;
                    auto oldIter = workingParts[checkingDim][i].first;
                    const Point* currentPoint = lastElements[i];
                    if (currentPoint == nullptr)
                    {
                        continue;
                    }
                    const auto& currentCoord = currentPoint->coord;
                    while (oldIter != workingParts[checkingDim][i].second)
                    {
                        const Point* oldPoint = *oldIter;
                        if (oldPoint != currentPoint)
                        {
                            if((oldPoint->coord)[lastDim] <= currentCoord[lastDim])
                            {
                                *currentIterFirst = *oldIter;
                                ++currentIterFirst;
                            }
                            else
                            {
                                *currentIterSecond = *oldIter;
                                ++currentIterSecond;
                            }
                        }
                        ++oldIter;
                    }
                    noPointsAdded &= (currentIterFirst == firstBegin && currentIterSecond == secondBegin);
                }

                std::swap(tempPointPointers, pointPointers[checkingDim]);
                std::swap(tempWorkingParts, workingParts[checkingDim]);
            }

            // Split the last dim
            {
                auto& tempWorkingParts = workingParts.back();
                tempWorkingParts.clear();

                for (const auto& workingPart : workingParts[lastDim])
                {
                    const auto& begin = workingPart.first;
                    const auto& end = workingPart.second;
                    if (begin != end)
                    {
                        auto median = (begin + (end - begin - 1) / 2);
                        tempWorkingParts.emplace_back(begin, median);
                        tempWorkingParts.emplace_back(median + 1, end);
                    }
                    else
                    {
                        tempWorkingParts.emplace_back(begin, end);
                    }
                }
                std::swap(tempWorkingParts, workingParts[lastDim]);

            }
        }
        if (noPointsAdded)
        {
            break;
        }

        currentDim = nextDim;
        nextDim = (nextDim + 1) % DIM;
        lastElements.clear();

        for (const auto& workingPart : workingParts[currentDim])
        {
            const auto& begin = workingPart.first;
            const auto& end = workingPart.second;
            if (begin != end)
            {
                auto median = (begin + (end - begin - 1) / 2);
                const Point* const point = *median;
                const auto& coordinates = point->coord;
                lastElements.emplace_back(point);
                tree->Insert(coordinates, point->id);
            }
            else
            {
                lastElements.emplace_back(nullptr);
            }
        }
    }

}

// Euclidian clustering implementation
template<typename PointT>
std::vector<PointCloudPtrTemplate<PointT>> ProcessPointClouds<PointT>::Clustering(const PointCloudPtr& cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    const int N{static_cast<int>(cloud->points.size())};
    KdTree tree{};
    FillKdTree(&tree, cloud);

    std::vector<bool> processedPoints(N, false);

    std::function<void(int, std::vector<int>&)> proximityFunc = [&] (int pointId, std::vector<int>& cluster)
    {
        const auto& point = cloud->points[static_cast<std::size_t>(pointId)];
        processedPoints[static_cast<std::size_t>(pointId)] = true;
        cluster.push_back(pointId);
        auto nearbyPoints = tree.Search({point.x, point.y, point.z}, clusterTolerance);
        for (int nearPointId: nearbyPoints)
        {
            if (processedPoints[static_cast<std::size_t>(nearPointId)] == false)
            {
                proximityFunc(nearPointId, cluster);
            }
        }
    };

    std::vector<pcl::PointIndices> clustersIndices;
    for (int pointId{0}; pointId < N; ++pointId)
    {
        if (processedPoints[static_cast<std::size_t>(pointId)] == false)
        {
            std::vector<int> newCluster;
            proximityFunc(pointId, newCluster);
            const int clusterSize{static_cast<int>(newCluster.size())};
            if (clusterSize >= minSize && clusterSize <= maxSize)
            {
                clustersIndices.emplace_back();
                clustersIndices.back().indices = std::move(newCluster);
            }
        }
    }

    std::vector<PointCloudPtr> clusters;
    clusters.reserve(clustersIndices.size());
    pcl::ExtractIndices<PointT> extract;
    for (auto& clusterIds : clustersIndices)
    {
        extract.setInputCloud(cloud);
        extract.setIndices(boost::shared_ptr<const pcl::PointIndices>(boost::shared_ptr<pcl::PointIndices>(), &clusterIds));
        extract.setNegative(false);
        clusters.push_back(boost::make_shared<pcl::PointCloud<PointT>>());
        extract.filter(*clusters.back());

        assert(clusters.back()->width == clusters.back()->points.size());
        assert(clusters.back()->height == 1);
        assert(clusters.back()->is_dense == true);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clustersIndices.size() << " clusters" << std::endl;

    return clusters;
}


template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(const PointCloudPtr& cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}


template<typename PointT>
void ProcessPointClouds<PointT>::SavePcd(const PointCloudPtr& cloud, const std::string& file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to "+file << std::endl;
}


template<typename PointT>
PointCloudPtrTemplate<PointT> ProcessPointClouds<PointT>::LoadPcd(const std::string& file)
{

    PointCloudPtr cloud{boost::make_shared<pcl::PointCloud<PointT>>()};

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size () << " data points from "+file << std::endl;

    return cloud;
}


template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::StreamPcd(const std::string& dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    std::sort(paths.begin(), paths.end());

    return paths;

}


#endif /* PROCESSPOINTCLOUDS_HPP_ */
