#ifndef PROCESSPOINTCLOUDS_HPP_
#define PROCESSPOINTCLOUDS_HPP_

#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/transforms.h>
#include <iostream> 
#include <string>  
#include <vector>
#include <ctime>
#include <cstddef>
#include <chrono>
#include <functional>
#include <random>
#include <unordered_set>

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
    cropBox.filter(filteredIndices.indices);
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloudCropped);
    extract.setIndices(boost::shared_ptr<const pcl::PointIndices>(boost::shared_ptr<pcl::PointIndices>(), &filteredIndices));
    extract.setNegative(true);
    extract.filter(*cloudCropped);

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
            const auto& point = cloud->points[static_cast<size_t>(j)];
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

// Euclidian clustering implementation
template<typename PointT>
std::vector<PointCloudPtrTemplate<PointT>> ProcessPointClouds<PointT>::Clustering(const PointCloudPtr& cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    const int N{static_cast<int>(cloud->points.size())};
    KdTree tree{};
    for (int i{0}; i < N; ++i)
    {
        const auto& point = cloud->points[static_cast<size_t>(i)];
        tree.Insert({point.x, point.y, point.z}, i);
    }

    std::vector<bool> processedPoints(N, false);

    std::function<void(int, std::vector<int>&)> proximityFunc = [&] (int pointId, std::vector<int>& cluster)
    {
        const auto& point = cloud->points[static_cast<size_t>(pointId)];
        processedPoints[static_cast<size_t>(pointId)] = true;
        cluster.push_back(pointId);
        auto nearbyPoints = tree.Search({point.x, point.y, point.z}, clusterTolerance);
        for (int nearPointId: nearbyPoints)
        {
            if (processedPoints[static_cast<size_t>(nearPointId)] == false)
            {
                proximityFunc(nearPointId, cluster);
            }
        }
    };

    std::vector<pcl::PointIndices> clustersIndices;
    for (int pointId{0}; pointId < N; ++pointId)
    {
        if (processedPoints[static_cast<size_t>(pointId)] == false)
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
