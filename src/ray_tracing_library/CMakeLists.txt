cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(ray_tracing_library)

add_library(${PROJECT_NAME} STATIC src/Renderer.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PUBLIC ../math_library/include)
target_include_directories(${PROJECT_NAME} PUBLIC ../stb_library/include)