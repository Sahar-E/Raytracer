cmake_minimum_required(VERSION 3.8.12)
set(CMAKE_CXX_STANDARD 20)
project(ray_tracer LANGUAGES CXX CUDA)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# CUDA:
FIND_PACKAGE(CUDA REQUIRED)
set(CUDA_ENABLE 1)


## OpenMP:
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#find_package(OpenMP REQUIRED)


add_executable(${PROJECT_NAME} main.cu temp.cpp)

# CUDA:
set(CMAKE_CUDA_STANDARD 14)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${PROJECT_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --relocatable-device-code=true
        >)

# UNIX, WIN32, WINRT, CYGWIN, APPLE are environment variables as flags set by default system
if(UNIX)
    message("This is a ${CMAKE_SYSTEM_NAME} system")
elseif(WIN32)
    message("This is a Windows System")
    message("WINDOWS FLAGS: ${CMAKE_CXX_FLAGS}")
#    macro(GMX_SET_CUDA_NVCC_FLAGS)
#        set(CUDA_NVCC_FLAGS "${GMX_CUDA_NVCC_FLAGS};${CUDA_NVCC_FLAGS};-lineinfo")
#    endmacro()
#    SET( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIBS:LIBCMT" )
#    target_link_options(${PROJECT_NAME} PUBLIC /INCREMENTAL:NO /NODEFAULTLIB:LIBCMT)
endif()


# Libraries:
include_directories(src/dependencies/)

add_subdirectory(src/dependencies/stb_library)
target_link_libraries(${PROJECT_NAME} stb_library)

add_subdirectory(src/ray_tracing_library)
target_link_libraries(${PROJECT_NAME} ray_tracing_library)

add_subdirectory(src/openGL)
target_link_libraries(${PROJECT_NAME} gui)

add_subdirectory(src/application)
target_link_libraries(${PROJECT_NAME} application)



find_package(OpenGL REQUIRED)
add_subdirectory(src/dependencies/glfw-3.3.7)
target_link_libraries(${PROJECT_NAME} glfw OpenGL::GL)

add_subdirectory(src/dependencies/glew-2.1.0/build/cmake)
target_link_libraries(${PROJECT_NAME} glew_s)

add_subdirectory(src/dependencies/imgui-docking)
target_link_libraries(${PROJECT_NAME} imgui)

add_subdirectory(src/dependencies/glm-0.9.9.8)
target_link_libraries(${PROJECT_NAME} glm)