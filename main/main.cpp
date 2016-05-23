#include <vector>
#include <string>
#include <fstream>

#include <vulkan/vk_cpp.hpp>
#include <glfw/glfw3.h>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>

// vulkan validation layers
const std::vector<const char*> validationLayers =
{
	"VK_LAYER_LUNARG_standard_validation",
};


// Helper entry function to create debug reports.
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugReportCallbackEXT(
    VkInstance                                  instance,
    const VkDebugReportCallbackCreateInfoEXT*   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDebugReportCallbackEXT*                   pCallback)
{
    static PFN_vkCreateDebugReportCallbackEXT createDebugReportCallbackEXT = nullptr;

    if (createDebugReportCallbackEXT == nullptr) {
        createDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)
            vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
        if (createDebugReportCallbackEXT == nullptr) {
            printf("Could not find vkCreateDebugReportCallbackEXT.");
            assert(!"error.");
            return (VkResult) 1;
        }
    }
 
    return createDebugReportCallbackEXT(instance, pCreateInfo, pAllocator, pCallback);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDebugReportCallbackEXT(
    VkInstance                                  instance,
    VkDebugReportCallbackEXT                    callback,
    const VkAllocationCallbacks*                pAllocator)
{
    static PFN_vkDestroyDebugReportCallbackEXT destroyDebugReportCallbackEXT = nullptr;

    if (destroyDebugReportCallbackEXT == nullptr) {
        destroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)
            vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
        if (destroyDebugReportCallbackEXT == nullptr) {
            printf("Could not find vkDestroyDebugReportCallbackEXT.");
            assert(!"error.");
            return;
        }
    }

    destroyDebugReportCallbackEXT(instance, callback, pAllocator);
}

uint32_t displayWidth = 640;
uint32_t displayHeight = 480;

const uint32_t sVertexBufferBindId = 0;

int main()
{
	glfwInit();

	if (!glfwVulkanSupported())
	{
		throw std::runtime_error("!glfwVulkanSupported");
	}

	vk::Result result;

	// Check out validation layers
	{
		uint32_t layerCount;
		if ((result = vk::enumerateInstanceLayerProperties(&layerCount, nullptr)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::enumerateInstanceLayerProperties -> " + std::to_string((int)result));
		}
		std::vector<vk::LayerProperties> layers(layerCount);
		if ((result = vk::enumerateInstanceLayerProperties(&layerCount, layers.data())) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::enumerateInstanceLayerProperties -> " + std::to_string((int)result));
		}

		for (auto layer : validationLayers)
		{
			bool found = false;
			for (auto layerProp : layers)
			{
				if (!strcmp(layerProp.layerName, layer))
				{
					found = true;
					break;
				}
			}
			if (!found) throw std::runtime_error("Validation layer not found : " + std::string(layer));
		}
	}

	// Setup vulkan instance
	vk::Instance vulkanInstance;
	{
		vk::ApplicationInfo applicationInfo(
			"vkTriangle", 1, // app name,version
			"vkTriangleEngine", 1, // engine name,version
			VK_MAKE_VERSION(1, 0, 2)); // vulkan api version

		// GLFW gets us extensions we need to support display
		int count;
		const char** glfwExts = glfwGetRequiredInstanceExtensions(&count);
		std::vector<const char*> extensions(glfwExts, glfwExts + count);
		if (validationLayers.size() > 0)
		{
			extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		}
		vk::InstanceCreateInfo instanceCreateInfo(vk::InstanceCreateFlags(), &applicationInfo,
			(uint32_t)validationLayers.size(), validationLayers.data(), // layers
			(uint32_t)extensions.size(), extensions.data()); // extensions

		//@todo check extensions exist
		if ((result = vk::createInstance(&instanceCreateInfo, nullptr, &vulkanInstance)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::createInstance -> " + std::to_string((int)result));
		}

		if (validationLayers.size() > 0)
		{
			auto debugCallback = [](
				VkDebugReportFlagsEXT flags,
				VkDebugReportObjectTypeEXT objType,
				uint64_t srcObject,
				size_t location,
				int32_t msgCode,
				const char* pLayerPrefix,
				const char* pMsg,
				void* pUserData)
				-> VkBool32
			{
				std::string msg(pMsg);
                printf("DEBUG LAYER: %s\n", msg.c_str());
				if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
				{
					throw std::runtime_error("debugCallback got error");
				}
				else if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT)
				{
					throw std::runtime_error("debugCallback got warning");
				}

				return false;
			};

			auto createDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT) vulkanInstance.getProcAddr("vkCreateDebugReportCallbackEXT");

			vk::DebugReportCallbackCreateInfoEXT debugInfo(
				vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning,
				debugCallback, nullptr);
			vk::DebugReportCallbackEXT debugReportCallback;
			if (vulkanInstance.createDebugReportCallbackEXT(&debugInfo, nullptr, &debugReportCallback) != vk::Result::eSuccess)
			{
				throw std::runtime_error("vk::createDebugReportCallbackEXT -> " + std::to_string((int)result));
			}
		}
	}

	// Get physical device
	vk::PhysicalDevice physicalDevice;
	vk::PhysicalDeviceMemoryProperties physDevMemProps;
	{
		uint32_t physCount = 0;
		vulkanInstance.enumeratePhysicalDevices(&physCount, nullptr);
		std::vector<vk::PhysicalDevice> physDevices(physCount);
		vulkanInstance.enumeratePhysicalDevices(&physCount, physDevices.data());
		if (physCount == 0)
		{
			throw std::runtime_error("vulkan: no physical devices");
		}

		physicalDevice = physDevices[0]; // use the first device, hardcode if you want to change

		// print device name
		vk::PhysicalDeviceProperties physProps;
		physicalDevice.getProperties(&physProps);
		printf("Device name: '%s'\n", physProps.deviceName);

        physicalDevice.getMemoryProperties(&physDevMemProps);
	}

	// Lambda function to poll device memory properties for a memory type we want, see vulkan spec 10.2: Device memory
	auto findDeviceMemoryType = [&](const vk::MemoryRequirements& memReqs, vk::MemoryPropertyFlags propReqs) {
		uint32_t memoryTypeIndex = physDevMemProps.memoryTypeCount;
		for (uint32_t i = 0; i < physDevMemProps.memoryTypeCount; i++)
		{
			// if i'th bit is enabled in MemoryRequirements AND memory property is DEVICE_LOCAL, i.e efficient for the device to access depth stencil.
			if ((memReqs.memoryTypeBits & (1 << i)) && (physDevMemProps.memoryTypes[i].propertyFlags & propReqs) == propReqs)
			{
				memoryTypeIndex = i;
				break;
			}
		}
		if (memoryTypeIndex >= physDevMemProps.memoryTypeCount)
		{
			throw std::runtime_error("No device memory types meet given requirements.");
		}
		return memoryTypeIndex;
	};

	// Create vulkan device from physical
	vk::Device vDevice;
	uint32_t graphicsQueueFamily;
	vk::Queue graphicsQueue;
	{
		// device extensions
		std::vector<const char*> enabledExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

		// get the graphics queue off the physical device
		{
			uint32_t queueCount;
            physicalDevice.getQueueFamilyProperties(&queueCount, NULL);
			std::vector<vk::QueueFamilyProperties> queueProps(queueCount);
            physicalDevice.getQueueFamilyProperties(&queueCount, queueProps.data());

			for (graphicsQueueFamily = 0; graphicsQueueFamily < queueCount; graphicsQueueFamily++)
			{
				// queue must support VK_QUEUE_GRAPHICS_BIT
				if (queueProps[graphicsQueueFamily].queueFlags & vk::QueueFlagBits::eGraphics)
				{
					// queue must also be compatible with glfw presentation requirements
					if (glfwGetPhysicalDevicePresentationSupport(vulkanInstance, physicalDevice, graphicsQueueFamily))
					{
						break;
					}
				}
			}
			if (graphicsQueueFamily >= queueCount)
			{
				throw std::runtime_error("physical device has no compatible graphics queue");
			}
		}

		std::vector<float> queuePriorities = { 0.0 };
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), graphicsQueueFamily, (uint32_t)queuePriorities.size(), queuePriorities.data());

		vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(), 1, &deviceQueueCreateInfo,
			(uint32_t)validationLayers.size(), validationLayers.data(), // validation layers
			(uint32_t)enabledExtensions.size(), enabledExtensions.data(), // extensions
			nullptr); // phys device features

		if ((result = physicalDevice.createDevice(&deviceCreateInfo, nullptr, &vDevice)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::createDevice -> " + std::to_string((int)result));
		}

		vDevice.getQueue(graphicsQueueFamily, 0, &graphicsQueue);
	}
	
	// Setup interop with windowing system, GLFW handles it
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // tell glfw vulkan runs the show
	GLFWwindow* window = glfwCreateWindow(displayWidth, displayHeight, "vkTriangle", NULL, NULL); // create window
	
	vk::SurfaceKHR surface;
    VkSurfaceKHR surface_ = (VkSurfaceKHR) surface;
	{
		if (glfwCreateWindowSurface(vulkanInstance, window, NULL, &(VkSurfaceKHR) surface_) != VK_SUCCESS)
		{
			throw std::runtime_error("glfwCreateWindowSurface failure");
		}
        surface = surface_;

		// double check GLFW's surface
		vk::Bool32 supported;
		if ((result = physicalDevice.getSurfaceSupportKHR(graphicsQueueFamily, surface, &supported)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::getPhysicalDeviceSurfaceSupportKHR -> " + std::to_string((int)result));
		}
		if (!supported)
		{
			throw std::runtime_error("Surface not supported, vkGetPhysicalDeviceSurfaceSupportKHR is false");
		}
	}

	// Get a format supported by the display surface
	vk::Format surfaceFormat;
	vk::ColorSpaceKHR surfaceColorSpace;
	{
		uint32_t surfaceFormatCount;
		physicalDevice.getSurfaceFormatsKHR(surface, &surfaceFormatCount, nullptr);
		std::vector<vk::SurfaceFormatKHR> surfaceFormats(surfaceFormatCount);
		physicalDevice.getSurfaceFormatsKHR(surface, &surfaceFormatCount, surfaceFormats.data());
		if (surfaceFormatCount == 1 && surfaceFormats[0].format == vk::Format::eUndefined)
		{
			surfaceFormat = vk::Format::eB8G8R8Unorm;
		}
		else if (surfaceFormatCount > 0)
		{
			surfaceFormat = surfaceFormats[0].format;
		}
		else
		{
			throw std::runtime_error("getPhysicalDeviceSurfaceFormatsKHR failure");
		}
		surfaceColorSpace = surfaceFormats[0].colorSpace;
	}

	// Command pool
	vk::CommandPool commandPool;
	{
		vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, graphicsQueueFamily);
		if ((result = vDevice.createCommandPool(&commandPoolCreateInfo, nullptr, &commandPool)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::createCommandPool -> " + std::to_string((int)result));
		}
	}

	// Swapchain setup
	vk::SwapchainKHR swapchain;
	{
		vk::SurfaceCapabilitiesKHR surfaceCaps;
		physicalDevice.getSurfaceCapabilitiesKHR(surface, &surfaceCaps);

		uint32_t presentModeCount;
		physicalDevice.getSurfacePresentModesKHR(surface, &presentModeCount, nullptr);
		std::vector<vk::PresentModeKHR> presentModes;
		presentModes.resize(presentModeCount);
		physicalDevice.getSurfacePresentModesKHR(surface, &presentModeCount, presentModes.data());

		vk::Extent2D swapchainExtent;
		// width and height are either both -1, or both not -1.
		if (surfaceCaps.currentExtent.width == -1)
		{
			// If the surface size is undefined, the size is set to
			// the size of the images requested, which must fit within the minimum
			// and maximum values.
			swapchainExtent.width = displayWidth;
			swapchainExtent.height = displayHeight;

			if (swapchainExtent.width < surfaceCaps.minImageExtent.width)
				swapchainExtent.width = (surfaceCaps.minImageExtent.width);
			else if (swapchainExtent.width > surfaceCaps.maxImageExtent.width)
				swapchainExtent.width = (surfaceCaps.maxImageExtent.width);

			if (swapchainExtent.height < surfaceCaps.minImageExtent.height)
				swapchainExtent.height = (surfaceCaps.minImageExtent.height);
			else if (swapchainExtent.height > surfaceCaps.maxImageExtent.height)
				swapchainExtent.height = (surfaceCaps.maxImageExtent.height);
		}
		else
		{
			// If the surface size is defined, the swap chain size must match
			swapchainExtent = surfaceCaps.currentExtent;
		}
		displayWidth = surfaceCaps.currentExtent.width;
		displayHeight = surfaceCaps.currentExtent.height;

		uint32_t swapchainImages = surfaceCaps.minImageCount + 1;
		if ((surfaceCaps.maxImageCount > 0) && (swapchainImages > surfaceCaps.maxImageCount))
		{
			swapchainImages = surfaceCaps.maxImageCount;
		}

		vk::SwapchainCreateInfoKHR swapCreateInfo(vk::SwapchainCreateFlagsKHR(), surface, swapchainImages,
			surfaceFormat, surfaceColorSpace, swapchainExtent,
			1, vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive,
			0, nullptr,
			surfaceCaps.currentTransform,
			vk::CompositeAlphaFlagBitsKHR::eOpaque, // disable alpha in compositing
			vk::PresentModeKHR::eFifo, // fifo for vsync on
			VK_TRUE,
			VK_NULL_HANDLE);
		vDevice.createSwapchainKHR(&swapCreateInfo, nullptr, &swapchain);
	}
	
	// setup vulkan image views into swapchain buffers
	struct SwapChainImage {
		VkImage image; // vulkan image
		VkImageView view; // view into the image
	};
	std::vector<SwapChainImage> swapChainViews;
	{
		uint32_t swapchainImageCount;
		vDevice.getSwapchainImagesKHR(swapchain, &swapchainImageCount, nullptr);
		std::vector<vk::Image> swapchainImages(swapchainImageCount);
		vDevice.getSwapchainImagesKHR(swapchain, &swapchainImageCount, swapchainImages.data());

		swapChainViews.resize(swapchainImageCount);

		for (uint32_t i = 0; i < swapchainImageCount; i++)
		{
			vk::ImageViewCreateInfo viewCreateInfo(vk::ImageViewCreateFlags(), swapchainImages[i], vk::ImageViewType::e2D, surfaceFormat,
				vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA),
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

            vk::ImageView view_(swapChainViews[i].view);
			if ((result = vDevice.createImageView(&viewCreateInfo, nullptr, &view_)) != vk::Result::eSuccess)
			{
				throw std::runtime_error("vk::createImageView -> " + std::to_string((int)result));
			}
			swapChainViews[i].image = swapchainImages[i];
            swapChainViews[i].view = view_;
		}
	}

	// setup depth stencil
	vk::Format depthStencilFormat;
	vk::Image depthStencilImage;
	vk::DeviceMemory depthStencilMemory;
	vk::ImageView depthStencilView;
	{
		// get a format for depth stencil supported by hardware
		std::vector<vk::Format> preferredFormatList = {
			vk::Format::eD32SfloatS8Uint,
			vk::Format::eD32Sfloat,
			vk::Format::eD24UnormS8Uint,
			vk::Format::eD16UnormS8Uint,
			vk::Format::eD16Unorm,
		};
		depthStencilFormat = vk::Format::eUndefined;
		for (auto format : preferredFormatList)
		{
			vk::FormatProperties formatProperties;
			physicalDevice.getFormatProperties(format, &formatProperties);
			if (formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
			{
				depthStencilFormat = format;
				break;
			}
		}
		if (depthStencilFormat == vk::Format::eUndefined)
		{
			throw std::runtime_error("no compatible depth stencil formats");
		}

		vk::ImageCreateInfo imgCreateInfo(vk::ImageCreateFlags(), vk::ImageType::e2D, depthStencilFormat,
			vk::Extent3D(displayWidth, displayHeight, 1), 1, 1,
			vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransferSrc,
			vk::SharingMode::eExclusive, 0, nullptr, vk::ImageLayout::eUndefined);
		if ((result = vDevice.createImage(&imgCreateInfo, nullptr, &depthStencilImage)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::createImage -> " + std::to_string((int)result));
		}
		
		vk::MemoryRequirements memoryReqs;
		vDevice.getImageMemoryRequirements(depthStencilImage, &memoryReqs);

		// find memory fitting memoryReqs and local to the device, for efficient access
		uint32_t memoryTypeIndex = findDeviceMemoryType(memoryReqs, vk::MemoryPropertyFlagBits::eDeviceLocal);

		vk::MemoryAllocateInfo memAllocInfo(memoryReqs.size, memoryTypeIndex);
		if ((result = vDevice.allocateMemory(&memAllocInfo, nullptr, &depthStencilMemory)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::allocateMemory -> " + std::to_string((int)result));
		}

		vDevice.bindImageMemory(depthStencilImage, depthStencilMemory, 0);

		vk::ImageViewCreateInfo imgViewCreateInfo(vk::ImageViewCreateFlags(), depthStencilImage, vk::ImageViewType::e2D,
			depthStencilFormat, vk::ComponentMapping(/*identity map*/),
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1));
		if ((result = vDevice.createImageView(&imgViewCreateInfo, nullptr, &depthStencilView)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::createImageView -> " + std::to_string((int)result));
		}
	}
	
	// renderpass setup
	vk::RenderPass renderPass;
	{
		vk::AttachmentDescription attachDescs[2] = {
			// rgb surface attachment
			vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), surfaceFormat, vk::SampleCountFlagBits::e1,
				vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
				vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
				vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal),
			// depth stencil attachment
			vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), depthStencilFormat, vk::SampleCountFlagBits::e1,
				vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
				vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
				vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal),
		};
		vk::AttachmentReference surfReference(0, vk::ImageLayout::eColorAttachmentOptimal);
		vk::AttachmentReference depthReference(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
		vk::SubpassDescription subpassDesc(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics,
			0, nullptr, 1, &surfReference, nullptr, &depthReference, 0, nullptr);
		vk::RenderPassCreateInfo renderPassCreateInfo(vk::RenderPassCreateFlags(), 2, attachDescs, 1, &subpassDesc, 0, nullptr);
		if ((result = vDevice.createRenderPass(&renderPassCreateInfo, nullptr, &renderPass)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::createRenderPass -> " + std::to_string((int)result));
		}
	}

	// pipeline cache setup
	vk::PipelineCache pipelineCache;
	{
		vk::PipelineCacheCreateInfo pipelineCreateInfo(vk::PipelineCacheCreateFlags(), 0, nullptr);
		if ((result = vDevice.createPipelineCache(&pipelineCreateInfo, nullptr, &pipelineCache)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::createPipelineCache -> " + std::to_string((int)result));
		}
	}

	// framebuffers setup
	std::vector<vk::Framebuffer> framebuffers(swapChainViews.size());
	{
		vk::ImageView attachments[2]; // color surface, depthstencil pair
		vk::FramebufferCreateInfo fbCreateInfo(vk::FramebufferCreateFlags(), renderPass, 2, attachments, displayWidth, displayHeight, 1);
		for (size_t i = 0; i < swapChainViews.size(); i++)
		{
			attachments[0] = swapChainViews[i].view;
			attachments[1] = depthStencilView;
			if ((result = vDevice.createFramebuffer(&fbCreateInfo, nullptr, &framebuffers[i])) != vk::Result::eSuccess)
			{
				throw std::runtime_error("vk::createFramebuffer -> " + std::to_string((int)result));
			}
		}
	}

	// initial image layout setups
	{
		// Create temp command buffer
		vk::CommandBuffer layoutCmdBuf;
		vk::CommandBufferAllocateInfo cbufAllocInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1);
		if ((result = vDevice.allocateCommandBuffers(&cbufAllocInfo, &layoutCmdBuf)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::allocateCommandBuffers -> " + std::to_string((int)result));
		}

		vk::CommandBufferBeginInfo cbufBegin;
		if ((result = layoutCmdBuf.begin(&cbufBegin)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::beginCommandBuffer -> " + std::to_string((int)result));
		}

		// transition swapchain images into presentation layout (VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
		for (const auto& swap : swapChainViews)
		{
			vk::ImageMemoryBarrier barrier(
				vk::AccessFlags(),
				vk::AccessFlags(),
				vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR,
				VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, swap.image,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
		    );
            layoutCmdBuf.pipelineBarrier(
				vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe,
				vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &barrier);
		}
		// transition depth stencil image (to VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		vk::ImageMemoryBarrier barrier(
			vk::AccessFlagBits(),
			vk::AccessFlagBits::eDepthStencilAttachmentWrite,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, depthStencilImage,
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1)
		);
		layoutCmdBuf.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe,
			vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &barrier);
        layoutCmdBuf.end();

		// submit
		vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &layoutCmdBuf, 0, nullptr);
		if ((result = graphicsQueue.submit(1, &submitInfo, VK_NULL_HANDLE)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::queueSubmit -> " + std::to_string((int)result));
		}
		graphicsQueue.waitIdle();

		vDevice.freeCommandBuffers(commandPool, 1, &layoutCmdBuf);
	}
	
	// setup commandbuffers for swapchain images, plus one extra for post-presentation barrier
	std::vector<vk::CommandBuffer> swapchainCmdBuffers(swapChainViews.size() + 1);
	vk::CommandBuffer presentBarrierCmdBuffer;
	{
		vk::CommandBufferAllocateInfo cmdBufAllocInfo(commandPool, vk::CommandBufferLevel::ePrimary, (uint32_t)swapchainCmdBuffers.size());
		if ((result = vDevice.allocateCommandBuffers(&cmdBufAllocInfo, swapchainCmdBuffers.data())) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::allocateCommandBuffers -> " + std::to_string((int)result));
		}
		presentBarrierCmdBuffer = swapchainCmdBuffers.back();
		swapchainCmdBuffers.pop_back();
	}

	// upload vertex data
	vk::Buffer vertexBuffer;
	vk::Buffer indexBuffer;
	{
		// Triangle vertex data
		std::vector<float> vertexData = {
			// (x,y,z,r,g,b)
			1, 1, 0, 1, 0, 0,
			-1, 1, 0, 0, 1, 0,
			0, -1, 0, 0, 0, 1,
		};
		std::vector<uint32_t> indexData = { 0, 1, 2 };

		// vertex
		{
			uint64_t vertexBytes = vertexData.size() * sizeof(decltype(vertexData)::value_type);

			vk::BufferCreateInfo bufCreateInfo(vk::BufferCreateFlags(), vertexBytes, vk::BufferUsageFlagBits::eVertexBuffer, vk::SharingMode::eExclusive, 0, nullptr);
			if ((result = vDevice.createBuffer(&bufCreateInfo, nullptr, &vertexBuffer)) != vk::Result::eSuccess)
			{
				throw std::runtime_error("vk::createBuffer -> " + std::to_string((int)result));
			}
			vk::MemoryRequirements memoryReqs;
			vDevice.getBufferMemoryRequirements(vertexBuffer, &memoryReqs);

			// find memory fitting memoryReqs and host visible
			uint32_t memoryTypeIndex = findDeviceMemoryType(memoryReqs, vk::MemoryPropertyFlagBits::eHostVisible);

			vk::DeviceMemory vBufMemory;
			vk::MemoryAllocateInfo memAllocInfo(memoryReqs.size, memoryTypeIndex);
			if ((result = vDevice.allocateMemory(&memAllocInfo, nullptr, &vBufMemory)) != vk::Result::eSuccess)
			{
				throw std::runtime_error("vk::allocateMemory -> " + std::to_string((int)result));
			}
			void* vBufRamData;
			vBufRamData = vDevice.mapMemory(vBufMemory, 0, memAllocInfo.allocationSize, vk::MemoryMapFlags());
			memcpy(vBufRamData, vertexData.data(), vertexBytes);
            vk::MappedMemoryRange range(vBufMemory, 0, memoryReqs.size);
            vDevice.flushMappedMemoryRanges(1, &range);
			vDevice.unmapMemory(vBufMemory);
			vDevice.bindBufferMemory(vertexBuffer, vBufMemory, 0);
		}
		// index
		{
			uint64_t indexBytes = indexData.size() * sizeof(decltype(indexData)::value_type);

			vk::BufferCreateInfo bufCreateInfo(vk::BufferCreateFlags(), indexBytes, vk::BufferUsageFlagBits::eIndexBuffer, vk::SharingMode::eExclusive, 0, nullptr);
			if ((result = vDevice.createBuffer(&bufCreateInfo, nullptr, &indexBuffer)) != vk::Result::eSuccess)
			{
				throw std::runtime_error("vk::createBuffer -> " + std::to_string((int)result));
			}
			vk::MemoryRequirements memoryReqs;
			vDevice.getBufferMemoryRequirements(indexBuffer, &memoryReqs);

			// find memory fitting memoryReqs and host visible
			uint32_t memoryTypeIndex = findDeviceMemoryType(memoryReqs, vk::MemoryPropertyFlagBits::eHostVisible);

			vk::DeviceMemory indexMemory;
			vk::MemoryAllocateInfo memAllocInfo(memoryReqs.size, memoryTypeIndex);
			if ((result = vDevice.allocateMemory(&memAllocInfo, nullptr, &indexMemory)) != vk::Result::eSuccess)
			{
				throw std::runtime_error("vk::allocateMemory -> " + std::to_string((int)result));
			}
			void* indexRamData;
			indexRamData = vDevice.mapMemory(indexMemory, 0, memAllocInfo.allocationSize, vk::MemoryMapFlagBits());
			if (!indexRamData) {
				throw std::runtime_error("vk::mapMemory -> " + std::to_string((int)result));
			}
			memcpy(indexRamData, indexData.data(), indexBytes);
            vk::MappedMemoryRange range(indexMemory, 0, memoryReqs.size);
            vDevice.flushMappedMemoryRanges(1, &range);
			vDevice.unmapMemory(indexMemory);
			vDevice.bindBufferMemory(indexBuffer, indexMemory, 0);
		}
	}

	// Uniform buffer object
	struct ubo {
		glm::mat4 projMatrix;
		glm::mat4 modelMatrix;
		glm::mat4 viewMatrix;
	} uboInstance;
	vk::Buffer uniformBuffer;
	vk::DeviceMemory uniformMemory;
	{
		vk::BufferCreateInfo bufCreateInfo(vk::BufferCreateFlags(), sizeof(ubo), vk::BufferUsageFlagBits::eUniformBuffer,
			vk::SharingMode::eExclusive, 0, nullptr);
		if ((result = vDevice.createBuffer(&bufCreateInfo, nullptr, &uniformBuffer)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::createBuffer -> " + std::to_string((int)result));
		}
		vk::MemoryRequirements memoryReqs;
		vDevice.getBufferMemoryRequirements(uniformBuffer, &memoryReqs);

		// find memory fitting memoryReqs and host visible
		uint32_t memoryTypeIndex = findDeviceMemoryType(memoryReqs, vk::MemoryPropertyFlagBits::eHostVisible);

		vk::MemoryAllocateInfo memAllocInfo(memoryReqs.size, memoryTypeIndex);
		if ((result = vDevice.allocateMemory(&memAllocInfo, nullptr, &uniformMemory)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::allocateMemory -> " + std::to_string((int)result));
		}

		// bind memory to buffer
		vDevice.bindBufferMemory(uniformBuffer, uniformMemory, 0);

		auto d2r = [](float degrees) {
			return degrees * 3.14159265358979323f / 180.f;
		};

		uboInstance.projMatrix = glm::perspective(d2r(60.0f), (float)displayWidth / (float)displayHeight, 0.1f, 256.0f);
		uboInstance.viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, -2.5f));

		uboInstance.modelMatrix = glm::mat4();
		uboInstance.modelMatrix = glm::rotate(uboInstance.modelMatrix, 0.f, glm::vec3(1.0f, 0.0f, 0.0f));
		uboInstance.modelMatrix = glm::rotate(uboInstance.modelMatrix, 0.f, glm::vec3(0.0f, 1.0f, 0.0f));
		uboInstance.modelMatrix = glm::rotate(uboInstance.modelMatrix, 0.f, glm::vec3(0.0f, 0.0f, 1.0f));

		// Map uniform buffer and update it
		uint8_t *pData = (uint8_t*) vDevice.mapMemory(uniformMemory, 0, sizeof(uboInstance), vk::MemoryMapFlags());
		if (!pData) {
			throw std::runtime_error("vk::mapMemory -> " + std::to_string((int)result));
		}
		memcpy(pData, &uboInstance, sizeof(uboInstance));
        vk::MappedMemoryRange range(uniformMemory, 0, sizeof(ubo));
        vDevice.flushMappedMemoryRanges(1, &range);
		vDevice.unmapMemory(uniformMemory);
	}
	
	// descriptor sets
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::PipelineLayout pipelineLayout;
	{
		vk::DescriptorSetLayoutBinding layoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);
		vk::DescriptorSetLayoutCreateInfo layoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), 1, &layoutBinding);
		if ((result = vDevice.createDescriptorSetLayout(&layoutCreateInfo, nullptr, &descriptorSetLayout)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::createDescriptorSetLayout -> " + std::to_string((int)result));
		}

		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), 1, &descriptorSetLayout, 0, nullptr);
		if ((result = vDevice.createPipelineLayout(&pipelineLayoutCreateInfo, nullptr, &pipelineLayout)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::createPipelineLayout -> " + std::to_string((int)result));
		}
	}
	
	// rendering pipeline state object
	vk::Pipeline pipeline;
	{
		// quick hack: VK_NV_glsl_shader allows glsl instead of SPIRV
		auto loadShader = [](const std::string& shader, vk::Device device)
		{
			vk::ShaderModuleCreateInfo shaderCreateInfo(vk::ShaderModuleCreateFlags(), shader.size(), (uint32_t*)shader.data());
			vk::ShaderModule shaderModule;
			vk::Result result = device.createShaderModule(&shaderCreateInfo, nullptr, &shaderModule);
			if (result != vk::Result::eSuccess)
			{
				throw std::runtime_error("vk::createShaderModule -> " + std::to_string((int)result));
			}
			return shaderModule;
		};

		std::ifstream vertFile("triangle.vert.spv", std::ios::in | std::ios::binary);
		std::string vertStr((std::istreambuf_iterator<char>(vertFile)), std::istreambuf_iterator<char>());
		std::ifstream fragFile("triangle.frag.spv", std::ios::in | std::ios::binary);
		std::string fragStr((std::istreambuf_iterator<char>(fragFile)), std::istreambuf_iterator<char>());

		vk::PipelineShaderStageCreateInfo shaderStages[] = {
			vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex,
				loadShader(vertStr, vDevice),
				"main", nullptr),
			vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment,
				loadShader(fragStr, vDevice),
				"main", nullptr),
		};

		vk::VertexInputBindingDescription vertexBindDesc(sVertexBufferBindId,
			6 * sizeof(float), // 6 float stride
			vk::VertexInputRate::eVertex);
		std::vector<vk::VertexInputAttributeDescription> vertexAttribDescs = {
			// position attribute (x,y,z)
			vk::VertexInputAttributeDescription(0, sVertexBufferBindId, vk::Format::eR32G32B32Sfloat, 0),
			// color attribute (r,g,b)
			vk::VertexInputAttributeDescription(1, sVertexBufferBindId, vk::Format::eR32G32B32Sfloat, 3 * sizeof(float)),
		};
		vk::PipelineVertexInputStateCreateInfo vertexInputCreate(vk::PipelineVertexInputStateCreateFlags(), 1, &vertexBindDesc, 
			(uint32_t)vertexAttribDescs.size(), vertexAttribDescs.data());

		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyStateCreate(vk::PipelineInputAssemblyStateCreateFlags(),
            vk::PrimitiveTopology::eTriangleList, false);

		// pViewports and pScissors is dynamic, therefore null here
		vk::PipelineViewportStateCreateInfo viewportStateCreate(vk::PipelineViewportStateCreateFlags(), 1, nullptr, 1, nullptr);

		vk::PipelineRasterizationStateCreateInfo rasterizationStateCreate(vk::PipelineRasterizationStateCreateFlags(), false, false,
			vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise,
			false, 0, 0, 0, 1.0f);

		vk::PipelineMultisampleStateCreateInfo multisampleStateCreate(vk::PipelineMultisampleStateCreateFlags(),
            vk::SampleCountFlagBits::e1, false, 0, nullptr, false, false);

		vk::PipelineDepthStencilStateCreateInfo depthStencilStateCreate(vk::PipelineDepthStencilStateCreateFlags(),
            true, true, vk::CompareOp::eLessOrEqual, false, false,
			vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways, 0, 0, 0),
			vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways, 0, 0, 0),
			0, 0);

		vk::PipelineColorBlendAttachmentState blendAttachment;
		vk::PipelineColorBlendStateCreateInfo colorBlendStateCreate(vk::PipelineColorBlendStateCreateFlags(),
            false, vk::LogicOp::eClear, 1, &blendAttachment, { 0 });

		std::vector<vk::DynamicState> dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
		vk::PipelineDynamicStateCreateInfo dynamicStateCreate(vk::PipelineDynamicStateCreateFlags(),
            (uint32_t)dynamicStates.size(), dynamicStates.data());

		vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo(
			vk::PipelineCreateFlags(),
			2, shaderStages,
			&vertexInputCreate,
			&inputAssemblyStateCreate,
			nullptr,
			&viewportStateCreate,
			&rasterizationStateCreate,
			&multisampleStateCreate,
			&depthStencilStateCreate,
			&colorBlendStateCreate,
			&dynamicStateCreate,
			pipelineLayout, renderPass,
			0, VK_NULL_HANDLE, 0);
		
		if ((result = vDevice.createGraphicsPipelines(pipelineCache, 1, &graphicsPipelineCreateInfo, nullptr, &pipeline)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::createGraphicsPipelines -> " + std::to_string((int)result));
		}
	}
	
	// descriptor pool
	vk::DescriptorPool descriptorPool;
	{
		vk::DescriptorPoolSize descPoolCounts[] = {
			vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
		};
		vk::DescriptorPoolCreateInfo descPoolCreate(vk::DescriptorPoolCreateFlags(), 1, 1, descPoolCounts);
		if ((result = vDevice.createDescriptorPool(&descPoolCreate, nullptr, &descriptorPool)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::createDescriptorPool -> " + std::to_string((int)result));
		}
	}

	// descriptor set
	vk::DescriptorSet descriptorSet;
	{
		vk::DescriptorSetAllocateInfo descAllocInfo(descriptorPool, 1, &descriptorSetLayout);
		if ((result = vDevice.allocateDescriptorSets(&descAllocInfo, &descriptorSet)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::allocateDescriptorSets -> " + std::to_string((int)result));
		}

		vk::DescriptorBufferInfo descBufferInfo(uniformBuffer, 0, sizeof(uboInstance));
		vk::WriteDescriptorSet writeDescriptorSet(descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &descBufferInfo, nullptr);
		vDevice.updateDescriptorSets(1, &writeDescriptorSet, 0, nullptr);
	}

	// swapchain command buffers, load with HelloTriangle
	{
		for (uint32_t i = 0; i < swapchainCmdBuffers.size(); i++)
		{
			auto& framebuffer = framebuffers[i];
			auto& commandBuf = swapchainCmdBuffers[i];
			auto& swapchainImage = swapChainViews[i].image;

			vk::CommandBufferBeginInfo cbufBegin;
			if ((result = commandBuf.begin(&cbufBegin)) != vk::Result::eSuccess)
			{
				throw std::runtime_error("vk::beginCommandBuffer -> " + std::to_string((int)result));
			}

            vk::ImageMemoryBarrier barrier1(
                vk::AccessFlagBits::eColorAttachmentWrite,
                vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
				vk::ImageLayout::ePresentSrcKHR, vk::ImageLayout::eColorAttachmentOptimal,
				VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, swapchainImage,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
            commandBuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTopOfPipe,
				vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &barrier1);

			vk::ClearValue clearValues[] = {
				vk::ClearValue(vk::ClearColorValue(std::array<float, 4>( { 0.4f, 0.4f, 0.4f, 1.f } ))),
				vk::ClearValue(vk::ClearColorValue(std::array<float, 4>( { 0.0f, 1.0f, 1.0f, 0.f } ))),
			};
			vk::RenderPassBeginInfo renderPassBegin(renderPass, framebuffer,
				vk::Rect2D(vk::Offset2D(), vk::Extent2D(displayWidth, displayHeight)),
				2, clearValues);
			commandBuf.beginRenderPass(&renderPassBegin, vk::SubpassContents::eInline);

			vk::Viewport viewport(0, 0, (float)displayWidth, (float)displayHeight, 0.f, 1.f);
			commandBuf.setViewport(0, 1, &viewport);

			vk::Rect2D scissor(vk::Offset2D(), vk::Extent2D(displayWidth, displayHeight));
			commandBuf.setScissor(0, 1, &scissor);

			commandBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
				pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

			commandBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

            vk::DeviceSize offset = 0;
			commandBuf.bindVertexBuffers(0, 1, &vertexBuffer, &offset);
			commandBuf.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);
			commandBuf.drawIndexed(3, 1, 0, 0, 1); // draw 3 verts out of index buffer

			commandBuf.endRenderPass();

			vk::ImageMemoryBarrier prePresentBarrier(vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlags(),
				vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
				VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, swapchainImage,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
			commandBuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTopOfPipe,
				vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &prePresentBarrier);

			commandBuf.end();
		}
	}

	// loop
	uint32_t swapchainBufferIndex;
	while (true)
	{
		//glfw
		
		vkDeviceWaitIdle(vDevice);

        vk::Semaphore presentationSem = vDevice.createSemaphore(vk::SemaphoreCreateInfo());

		if ((result = vDevice.acquireNextImageKHR(swapchain, UINT64_MAX, presentationSem, vk::Fence(), &swapchainBufferIndex)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::acquireNextImageKHR -> " + std::to_string((int)result));
		}
		vk::CommandBuffer cbuf = swapchainCmdBuffers[swapchainBufferIndex];
		vk::Image swapImage = swapChainViews[swapchainBufferIndex].image;

        vk::PipelineStageFlags pipeline = vk::PipelineStageFlagBits::eAllCommands;
		vk::SubmitInfo submitInfo(1, &presentationSem, &pipeline, 1, &cbuf, 0, nullptr);
		result = graphicsQueue.submit(1,
			&submitInfo,
			vk::Fence());
		if (result != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::queueSubmit -> " + std::to_string((int)result));
		}
		
		vk::PresentInfoKHR presentInfo(0, nullptr, 1, &swapchain, &swapchainBufferIndex, nullptr);
		if ((result = graphicsQueue.presentKHR(&presentInfo)) != vk::Result::eSuccess)
		{
			throw std::runtime_error("vk::queuePresentKHR -> " + std::to_string((int)result));
		}

        vDevice.waitIdle();
        vDevice.destroySemaphore(presentationSem, nullptr);

        static int frameN = 0;
        printf("frame %d\n", frameN++);
	}

	return 0;
}
