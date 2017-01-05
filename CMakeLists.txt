cmake_minimum_required(VERSION 3.6)
project(marsLander)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(marsLander ${SOURCE_FILES})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
find_package(SDL2 REQUIRED)
find_package(SDL2GFX REQUIRED)
include_directories(${SDL2_INCLUDE_DIR})
include_directories(${SDL2GFX_INCLUDE_DIR})
target_link_libraries(marsLander ${SDL2_LIBRARY})
target_link_libraries(marsLander ${SDL2GFX_LIBRARY_TEMP})