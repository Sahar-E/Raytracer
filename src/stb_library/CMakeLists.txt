cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(stb_library)

add_library(${PROJECT_NAME} STATIC src/stb.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include)