cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(gui LANGUAGES CXX)

# Library set up:
add_library(${PROJECT_NAME} STATIC
        include/commonOpenGL.h
        src/commonOpenGL.cpp
        src/GUIRenderer.cpp
        include/GUIRenderer.h
        src/VertexBuffer.cpp
        include/VertexBuffer.h
        src/IndexBuffer.cpp
        include/IndexBuffer.h
        src/VertexArray.cpp include/VertexArray.h src/VertexBufferLayout.cpp include/VertexBufferLayout.h src/Shader.cpp include/Shader.h)
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
