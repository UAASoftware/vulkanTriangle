cmake_minimum_required (VERSION 3.4)
project(vulkan-triangle)

include(externals.cmake)

add_subdirectory(glm)
add_subdirectory(glfw)
add_subdirectory(vkcpp)
add_subdirectory(main)