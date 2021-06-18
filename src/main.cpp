#include <iostream>
#include <cstddef>

#include "processPointClouds.hpp"
#include "render/render.h"

//setAngle: SWITCH CAMERA ANGLE {XY, TopDown, Side, FPS}
static void initCamera(CameraAngle setAngle, pcl::visualization::PCLVisualizer::Ptr& viewer)
{

    viewer->setBackgroundColor (0, 0, 0);

    // set camera position and angle
    viewer->initCameraParameters();
    // distance away in meters
    int distance = 16;

    switch(setAngle)
    {
        case XY : viewer->setCameraPosition(-distance, -distance, distance, 1, 1, 0); break;
        case TopDown : viewer->setCameraPosition(0, 0, distance, 1, 0, 1); break;
        case Side : viewer->setCameraPosition(0, -distance, 0, 0, 0, 1); break;
        case FPS : viewer->setCameraPosition(-10, 0, 0, 0, 0, 1);
    }

    if(setAngle!=FPS)
        viewer->addCoordinateSystem (1.0);
}

static void cityBlock(pcl::visualization::PCLVisualizer::Ptr& viewer, ProcessPointClouds<pcl::PointXYZI>& pointProcessorI, const pcl::PointCloud<pcl::PointXYZI>::Ptr& inputCloud)
{
    // Filter the cloud
    static const Eigen::Vector4f roofMin{-1.5f, -1.7f, -1, 1};
    static const Eigen::Vector4f roofMax{2.6f, 1.7f, 0.4f, 1};
    static const Eigen::Vector4f fieldOfViewMin{-12, -6, -2, 1};
    static const Eigen::Vector4f fieldOfViewMax{25, 6, 4, 1};
    static const float resolution{0.3f};
    const auto filteredCloud = pointProcessorI.FilterCloud(inputCloud, resolution , fieldOfViewMin, fieldOfViewMax, roofMin, roofMax);

    // Segment point cloud using RANSAC
    static const int ransacIterations{30};
    static const float ransacDistThreshold{0.3f};
    static const Color obstacleColor{1, 0.5f, 0}; // Orange
    static const Color planeColor{0, 1, 0};
    const auto segmentCloud = pointProcessorI.SegmentPlane(filteredCloud, ransacIterations, ransacDistThreshold);
    renderPointCloud(viewer, segmentCloud.first, "ObstCloud", obstacleColor);
    renderPointCloud(viewer, segmentCloud.second, "PlaneCloud", planeColor);

    // Perform Euclidian clustering
    static const float clusterTolerance{0.3f};
    static const int minClusterSize{6};
    static const int maxClusterSize{500};
    const auto cloudClusters = pointProcessorI.Clustering(segmentCloud.first, clusterTolerance, minClusterSize, maxClusterSize);

    std::size_t clusterId = 0;
    static const std::vector<Color> colors = {Color{1,0,0}, Color{1,1,0}, Color{0,0,1}};

    for(const auto& cluster : cloudClusters)
    {
          std::cout << "cluster size ";
          pointProcessorI.NumPoints(cluster);

          const auto& clusterColor = colors[clusterId % colors.size()];
          renderPointCloud(viewer, cluster, "obstCloud "+ std::to_string(clusterId), clusterColor);

          const Box box = pointProcessorI.BoundingBox(cluster);
          renderBox(viewer, box, static_cast<int>(clusterId), clusterColor);
          ++clusterId;
    }
}

int main (int argc, char** argv)
{
    const std::string dataPath = argc > 1 ? argv[1] : "../pcd/data_1";
    pcl::visualization::PCLVisualizer::Ptr viewer{boost::make_shared<pcl::visualization::PCLVisualizer>("3D Viewer")};
    initCamera(FPS, viewer);
    {
        ProcessPointClouds<pcl::PointXYZI> pointProcessorI;
        const std::vector<boost::filesystem::path> stream = pointProcessorI.StreamPcd(dataPath);
        auto streamIterator = stream.begin();
        pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloudI;

        while (!viewer->wasStopped ())
        {

          // Clear viewer
          viewer->removeAllPointClouds();
          viewer->removeAllShapes();

          // Load pcd and run obstacle detection process
          inputCloudI = pointProcessorI.LoadPcd(streamIterator->string());
          cityBlock(viewer, pointProcessorI, inputCloudI);

          streamIterator++;
          if(streamIterator == stream.end())
          {
            streamIterator = stream.begin();
          }

          viewer->spinOnce();
        }
    }
    return 0;
}
