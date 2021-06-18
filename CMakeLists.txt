cmake_minimum_required(VERSION 2.8)

project(lidar_detection)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

set(SOURCES_LIST
    src/main.cpp
    src/kdtree.cpp
    src/render/render.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCES_LIST})
target_link_libraries (${PROJECT_NAME} ${PCL_LIBRARIES})
