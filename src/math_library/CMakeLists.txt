cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(math_library)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(${PROJECT_NAME} STATIC src/Vec3.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include)