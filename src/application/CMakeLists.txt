cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(application LANGUAGES CXX)

# Library set up:
add_library(${PROJECT_NAME} STATIC
        include/Application.cuh
        src/Application.cu
        )
target_include_directories(${PROJECT_NAME} PUBLIC include)


# Libraries:
include_directories(src/dependencies/)

find_package(OpenGL REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ../dependencies/glfw-3.3.7)
target_link_libraries(${PROJECT_NAME} glfw OpenGL::GL)

target_include_directories(${PROJECT_NAME} PUBLIC ../dependencies/glew-2.1.0/build/cmake)
target_link_libraries(${PROJECT_NAME} glew_s)

target_include_directories(${PROJECT_NAME} PUBLIC ../dependencies/imgui-docking)
target_link_libraries(${PROJECT_NAME} imgui)

target_include_directories(${PROJECT_NAME} PUBLIC ../dependencies/glm-0.9.9.8)
target_link_libraries(${PROJECT_NAME} glm)

target_include_directories(${PROJECT_NAME} PUBLIC ../gui)
target_link_libraries(${PROJECT_NAME} gui)

target_include_directories(${PROJECT_NAME} PUBLIC ../ray_tracing_library)
target_link_libraries(${PROJECT_NAME} ray_tracing_library)


#
## CUDA:
#find_package(CUDA REQUIRED)
#set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#        --relocatable-device-code=true
#        --generate-line-info
#        #        --device-debug
#        >)
#
##if (WIN32)
##    macro(GMX_SET_CUDA_NVCC_FLAGS)
##        set(CUDA_NVCC_FLAGS "${GMX_CUDA_NVCC_FLAGS};${CUDA_NVCC_FLAGS};-lineinfo")
##    endmacro()
##endif()