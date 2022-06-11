cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(utils_library)
add_library(${PROJECT_NAME} STATIC src/utils.cpp include/constants.h)
target_include_directories(${PROJECT_NAME} PUBLIC include)

target_include_directories(${PROJECT_NAME} PUBLIC ../math_library)
target_link_libraries(${PROJECT_NAME} math_library)

target_include_directories(${PROJECT_NAME} PUBLIC ../stb_library)
target_link_libraries(${PROJECT_NAME} stb_library)
