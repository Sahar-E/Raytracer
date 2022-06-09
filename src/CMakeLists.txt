cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(math_library)
add_subdirectory(utils_library)
add_subdirectory(stb_library)
add_subdirectory(ray_tracing_library)

target_link_libraries(${PROJECT_NAME}
        math_library
        utils_library
        stb_library
        ray_tracing_library
        )