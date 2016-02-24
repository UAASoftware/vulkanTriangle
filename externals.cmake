set(EXT_VULKAN_SDK_INCLUDE "VulkanSDK/Include"
	CACHE PATH "Vulkan SDK include directory")
set(EXT_VULKAN_SDK_LIBS "VulkanSDK/Source/lib"
	CACHE PATH "Vulkan SDK lib directory")

add_library(vk_lunarg_layer_utils SHARED IMPORTED)
set_property(TARGET vk_lunarg_layer_utils PROPERTY IMPORTED_IMPLIB ${EXT_VULKAN_SDK_LIBS}/layer_utils.lib)
set_property(TARGET vk_lunarg_layer_utils PROPERTY IMPORTED_LOCATION ${EXT_VULKAN_SDK_LIBS}/layer_utils.dll)

add_library(vk_lunarg_vulkan SHARED IMPORTED)
set_property(TARGET vk_lunarg_vulkan PROPERTY IMPORTED_IMPLIB ${EXT_VULKAN_SDK_LIBS}/vulkan-1.lib)
set_property(TARGET vk_lunarg_vulkan PROPERTY IMPORTED_LOCATION ${EXT_VULKAN_SDK_LIBS}/vulkan-1.dll)

add_library(vk_lunarg INTERFACE)
target_include_directories(vk_lunarg INTERFACE ${EXT_VULKAN_SDK_INCLUDE})
target_link_libraries(vk_lunarg INTERFACE
	vk_lunarg_layer_utils
	vk_lunarg_vulkan
)
