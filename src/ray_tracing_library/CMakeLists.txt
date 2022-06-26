cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(ray_tracing_library)

# OpenMP:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
find_package(OpenMP REQUIRED)

add_library(${PROJECT_NAME} STATIC
        src/World.cu
        src/Camera.cu
        src/Renderer.cu
        include/Renderer.cuh
        src/Sphere.cu
        src/Material.cu
        include/Material.cuh
        src/Ray.cu
        include/Ray.cuh
        src/Vec3.cu
        include/Vec3.cuh
        src/utils.cpp
        include/constants.h)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --relocatable-device-code=true
        >)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_include_directories(${PROJECT_NAME} PUBLIC ../stb_library)
target_link_libraries(${PROJECT_NAME} stb_library)
