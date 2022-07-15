cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(imgui)









add_library(${PROJECT_NAME} STATIC
        src/imgui.cpp
        src/imgui_demo.cpp
        src/imgui_draw.cpp
        src/imgui_impl_glfw.cpp
        src/imgui_impl_opengl3.cpp
        src/imgui_tables.cpp
        src/imgui_widgets.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include)