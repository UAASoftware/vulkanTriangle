set(EXT_VULKAN_SDK_INCLUDE "VulkanSDK/Include"
	CACHE PATH "Vulkan SDK include directory")
set(EXT_VULKAN_SDK_LIBS "VulkanSDK/Source/lib"
	CACHE PATH "Vulkan SDK lib directory")

add_library(vulkan_lunarg SHARED IMPORTED GLOBAL)

set_property(TARGET vulkan_lunarg
	PROPERTY INTERFACE_INCLUDE_DIRECTORIES
	${EXT_VULKAN_SDK_INCLUDE})
set_property(TARGET vulkan_lunarg
	PROPERTY IMPORTED_IMPLIB
	${EXT_VULKAN_SDK_LIBS}/vulkan-1.lib)
set_property(TARGET vulkan_lunarg
	PROPERTY IMPORTED_LOCATION
	${EXT_VULKAN_SDK_LIBS}/vulkan-1.dll)
