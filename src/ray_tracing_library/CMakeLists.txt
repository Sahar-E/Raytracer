cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(ray_tracing_library)

add_library(${PROJECT_NAME} STATIC
        src/Renderer.cpp
        src/World.cpp
        src/Camera.cpp
        src/Sphere.cpp
        )

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_include_directories(${PROJECT_NAME} PUBLIC ../math_library)
target_link_libraries(${PROJECT_NAME} math_library)

target_include_directories(${PROJECT_NAME} PUBLIC ../stb_library)
target_link_libraries(${PROJECT_NAME} stb_library)

target_include_directories(${PROJECT_NAME} PUBLIC ../utils_library)
target_link_libraries(${PROJECT_NAME} utils_library)
