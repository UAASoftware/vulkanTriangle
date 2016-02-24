#include_directories()

add_executable(vkTriangle
  main.cpp
)

target_link_libraries(vkTriangle
  vk_lunarg
  vkcpp
  glfw
  glm
)
