cmake_minimum_required(VERSION 3.16)
project(ray_tracer)
set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# UNIX, WIN32, WINRT, CYGWIN, APPLE are environment variables as flags set by default system
if(UNIX)
    message("This is a ${CMAKE_SYSTEM_NAME} system")
elseif(WIN32)
    message("This is a Windows System")
endif()


# OpenMP:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
find_package(OpenMP REQUIRED)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#add_library(${PROJECT_NAME} STATIC main.cpp)
add_executable(${PROJECT_NAME} main.cpp)

add_subdirectory(src/math_library)
target_link_libraries(${PROJECT_NAME} math_library)

add_subdirectory(src/utils_library)
target_link_libraries(${PROJECT_NAME} utils_library)

add_subdirectory(src/stb_library)
target_link_libraries(${PROJECT_NAME} stb_library)

add_subdirectory(src/ray_tracing_library)
target_link_libraries(${PROJECT_NAME} ray_tracing_library)
