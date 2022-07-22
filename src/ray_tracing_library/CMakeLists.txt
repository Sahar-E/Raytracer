cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(ray_tracing_library LANGUAGES CXX CUDA)


# OpenMP:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
find_package(OpenMP REQUIRED)

# Library set up:
add_library(${PROJECT_NAME} STATIC
        include/World.cuh
        src/World.cu
        src/Camera.cu
        src/RayTracerRenderer.cu
        include/RayTracerRenderer.cuh
        src/Sphere.cu
        src/Material.cu
        include/Material.cuh
        src/Ray.cu
        include/Ray.cuh
        src/Vec3.cu
        include/Vec3.cuh
        src/my_math_cuda.cu
        include/my_math_cuda.cuh
        src/utils.cpp
        include/commonCuda.cuh
        src/commonCuda.cu
        include/my_math.h
        src/my_math.cpp
        include/commonDefines.h)

# CUDA:
find_package(CUDA REQUIRED)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --relocatable-device-code=true
        --generate-line-info
#        --device-debug
        >)

#if (WIN32)
#    macro(GMX_SET_CUDA_NVCC_FLAGS)
#        set(CUDA_NVCC_FLAGS "${GMX_CUDA_NVCC_FLAGS};${CUDA_NVCC_FLAGS};-lineinfo")
#    endmacro()
#endif()

# Include:
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PUBLIC ../dependencies/stb_library)
target_link_libraries(${PROJECT_NAME} stb_library)

target_include_directories(${PROJECT_NAME} PUBLIC ../dependencies/glm-0.9.9.8)
target_link_libraries(${PROJECT_NAME} glm)