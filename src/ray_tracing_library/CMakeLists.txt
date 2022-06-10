cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(ray_tracing_library)

add_library(${PROJECT_NAME} STATIC
        src/Renderer.cpp
        src/World.cpp
        src/Camera.cpp
        src/Sphere.cpp include/Sphere.h)

include_directories(PUBLIC include)
include_directories(PUBLIC ../math_library/include)
include_directories(PUBLIC ../stb_library/include)
include_directories(PUBLIC ../utils_library/include)