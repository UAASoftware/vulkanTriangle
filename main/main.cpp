#include <vector>
#include <string>
#include <fstream>

#include <vulkan/vk_cpp.h>
#include <glfw/glfw3.h>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>

// vulkan validation layers
const std::vector<const char*> validationLayers =
{
	//"VK_LAYER_GOOGLE_threading",
	//"VK_LAYER_LUNARG_device_limits",
	//"VK_LAYER_LUNARG_draw_state",
	//"VK_LAYER_LUNARG_image",
	//"VK_LAYER_LUNARG_mem_tracker",
	//"VK_LAYER_LUNARG_object_tracker",
	//"VK_LAYER_LUNARG_param_checker",
	//"VK_LAYER_LUNARG_swapchain",
	//"VK_LAYER_GOOGLE_unique_objects",
};

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

	// Setup vulkan instance
	vk::Instance vulkanInstance;
	{
		vk::ApplicationInfo applicationInfo(
			"vkTriangle", 1, // app name,version
			"vkTriangleEngine", 1, // engine name,version
			VK_MAKE_VERSION(1, 0, 2)); // vulkan api version

		// GLFW gets us extensions we need to support display
		int count;
		const char** extensions = glfwGetRequiredInstanceExtensions(&count);
		vk::InstanceCreateInfo instanceCreateInfo(0, &applicationInfo,
			(uint32_t)validationLayers.size(), validationLayers.data(), // layers
			count, extensions); // extensions

		//@todo check extensions exist
		if ((result = vk::createInstance(&instanceCreateInfo, nullptr, &vulkanInstance)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::createInstance -> " + std::to_string((int)result));
		}
	}

	// Get physical device
	vk::PhysicalDevice physicalDevice;
	vk::PhysicalDeviceMemoryProperties physDevMemProps;
	{
		uint32_t physCount = 0;
		vk::enumeratePhysicalDevices(vulkanInstance, &physCount, nullptr);
		std::vector<vk::PhysicalDevice> physDevices(physCount);
		vk::enumeratePhysicalDevices(vulkanInstance, &physCount, physDevices.data());
		if (physCount == 0)
		{
			throw std::runtime_error("vulkan: no physical devices");
		}

		physicalDevice = physDevices[0]; // use the first device, hardcode if you want to change

		// print device name
		vk::PhysicalDeviceProperties physProps;
		vk::getPhysicalDeviceProperties(physicalDevice, &physProps);
		printf("Device name: '%s'\n", physProps.deviceName());

		vk::getPhysicalDeviceMemoryProperties(physicalDevice, &physDevMemProps);
	}

	// Lambda function to poll device memory properties for a memory type we want, see vulkan spec 10.2: Device memory
	auto findDeviceMemoryType = [&](const vk::MemoryRequirements& memReqs, vk::MemoryPropertyFlags propReqs) {
		uint32_t memoryTypeIndex = physDevMemProps.memoryTypeCount();
		for (uint32_t i = 0; i < physDevMemProps.memoryTypeCount(); i++)
		{
			// if i'th bit is enabled in MemoryRequirements AND memory property is DEVICE_LOCAL, i.e efficient for the device to access depth stencil.
			if ((memReqs.memoryTypeBits() & (1 << i)) && (physDevMemProps.memoryTypes()[i].propertyFlags() & propReqs) == propReqs)
			{
				memoryTypeIndex = i;
				break;
			}
		}
		if (memoryTypeIndex >= physDevMemProps.memoryTypeCount())
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
		std::vector<const char*> enabledExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME, "VK_NV_glsl_shader" };

		// get the graphics queue off the physical device
		{
			uint32_t queueCount;
			vk::getPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, NULL);
			std::vector<vk::QueueFamilyProperties> queueProps(queueCount);
			vk::getPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, queueProps.data());

			for (graphicsQueueFamily = 0; graphicsQueueFamily < queueCount; graphicsQueueFamily++)
			{
				// queue must support VK_QUEUE_GRAPHICS_BIT
				if (queueProps[graphicsQueueFamily].queueFlags() & vk::QueueFlagBits::eGraphics)
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
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo(0, graphicsQueueFamily, (uint32_t)queuePriorities.size(), queuePriorities.data());

		vk::DeviceCreateInfo deviceCreateInfo(0, 1, &deviceQueueCreateInfo,
			(uint32_t)validationLayers.size(), validationLayers.data(), // validation layers
			(uint32_t)enabledExtensions.size(), enabledExtensions.data(), // extensions
			nullptr); // phys device features

		if ((result = vk::createDevice(physicalDevice, &deviceCreateInfo, nullptr, &vDevice)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::createDevice -> " + std::to_string((int)result));
		}

		vk::getDeviceQueue(vDevice, graphicsQueueFamily, 0, &graphicsQueue);
	}

	// Setup interop with windowing system, GLFW handles it
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // tell glfw vulkan runs the show
	GLFWwindow* window = glfwCreateWindow(displayWidth, displayHeight, "vkTriangle", NULL, NULL); // create window

	vk::SurfaceKHR surface;
	if (glfwCreateWindowSurface(vulkanInstance, window, NULL, &surface) != VK_SUCCESS)
	{
		throw std::runtime_error("glfwCreateWindowSurface failure");
	}

	// Get a format supported by the display surface
	vk::Format surfaceFormat;
	vk::ColorSpaceKHR surfaceColorSpace;
	{
		uint32_t surfaceFormatCount;
		vk::getPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &surfaceFormatCount, nullptr);
		std::vector<vk::SurfaceFormatKHR> surfaceFormats(surfaceFormatCount);
		vk::getPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &surfaceFormatCount, surfaceFormats.data());
		if (surfaceFormatCount == 1 && surfaceFormats[0].format() == vk::Format::eUndefined)
		{
			surfaceFormat = vk::Format::eB8G8R8Unorm;
		}
		else if (surfaceFormatCount > 0)
		{
			surfaceFormat = surfaceFormats[0].format();
		}
		else
		{
			throw std::runtime_error("getPhysicalDeviceSurfaceFormatsKHR failure");
		}
		surfaceColorSpace = surfaceFormats[0].colorSpace();
	}

	//// @todo optional, enable debug logging
	// console output
	//vkDebug::setupDebugging(instance, VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT, NULL);

	// Command pool
	vk::CommandPool commandPool;
	{
		vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, graphicsQueueFamily);
		if ((result = vk::createCommandPool(vDevice, &commandPoolCreateInfo, nullptr, &commandPool)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::createCommandPool -> " + std::to_string((int)result));
		}
	}

	// Swapchain setup
	vk::SwapchainKHR swapchain;
	{
		vk::SurfaceCapabilitiesKHR surfaceCaps;
		vk::getPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCaps);

		uint32_t presentModeCount;
		vk::getPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
		std::vector<vk::PresentModeKHR> presentModes;
		presentModes.resize(presentModeCount);
		vk::getPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());

		vk::Extent2D swapchainExtent;
		// width and height are either both -1, or both not -1.
		if (surfaceCaps.currentExtent().width() == -1)
		{
			// If the surface size is undefined, the size is set to
			// the size of the images requested, which must fit within the minimum
			// and maximum values.
			swapchainExtent.width(displayWidth);
			swapchainExtent.height(displayHeight);

			if (swapchainExtent.width() < surfaceCaps.minImageExtent().width())
				swapchainExtent.width(surfaceCaps.minImageExtent().width());
			else if (swapchainExtent.width() > surfaceCaps.maxImageExtent().width())
				swapchainExtent.width(surfaceCaps.maxImageExtent().width());

			if (swapchainExtent.height() < surfaceCaps.minImageExtent().height())
				swapchainExtent.height(surfaceCaps.minImageExtent().height());
			else if (swapchainExtent.height() > surfaceCaps.maxImageExtent().height())
				swapchainExtent.height(surfaceCaps.maxImageExtent().height());
		}
		else
		{
			// If the surface size is defined, the swap chain size must match
			swapchainExtent = surfaceCaps.currentExtent();
		}
		displayWidth = surfaceCaps.currentExtent().width();
		displayHeight = surfaceCaps.currentExtent().height();

		uint32_t swapchainImages = surfaceCaps.minImageCount() + 1;
		if ((surfaceCaps.maxImageCount() > 0) && (swapchainImages > surfaceCaps.maxImageCount()))
		{
			swapchainImages = surfaceCaps.maxImageCount();
		}

		vk::SwapchainCreateInfoKHR swapCreateInfo(0, surface, swapchainImages,
			surfaceFormat, surfaceColorSpace, swapchainExtent,
			1, vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive,
			0, nullptr,
			surfaceCaps.currentTransform(),
			vk::CompositeAlphaFlagBitsKHR::eOpaque, // disable alpha in compositing
			vk::PresentModeKHR::eVkPresentModeFifoKhr, // fifo for vsync on
			VK_TRUE,
			VK_NULL_HANDLE);
		vk::createSwapchainKHR(vDevice, &swapCreateInfo, nullptr, &swapchain);
	}

	// setup vulkan image views into swapchain buffers
	struct SwapChainImage {
		VkImage image; // vulkan image
		VkImageView view; // view into the image
	};
	std::vector<SwapChainImage> swapChainViews;
	{
		uint32_t swapchainImageCount;
		vk::getSwapchainImagesKHR(vDevice, swapchain, &swapchainImageCount, nullptr);
		std::vector<vk::Image> swapchainImages(swapchainImageCount);
		vk::getSwapchainImagesKHR(vDevice, swapchain, &swapchainImageCount, swapchainImages.data());

		swapChainViews.resize(swapchainImageCount);

		for (uint32_t i = 0; i < swapchainImageCount; i++)
		{
			vk::ImageViewCreateInfo viewCreateInfo(0, swapchainImages[i], vk::ImageViewType::e2D, surfaceFormat,
				vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA),
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

			if ((result = vk::createImageView(vDevice, &viewCreateInfo, nullptr, &swapChainViews[i].view)) != vk::Result::eVkSuccess)
			{
				throw std::runtime_error("vk::createImageView -> " + std::to_string((int)result));
			}
			swapChainViews[i].image = swapchainImages[i];
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
			vk::getPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);
			if (formatProperties.optimalTilingFeatures() & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
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
		if ((result = vk::createImage(vDevice, &imgCreateInfo, nullptr, &depthStencilImage)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::createImage -> " + std::to_string((int)result));
		}
		
		vk::MemoryRequirements memoryReqs;
		vk::getImageMemoryRequirements(vDevice, depthStencilImage, &memoryReqs);

		// find memory fitting memoryReqs and local to the device, for efficient access
		uint32_t memoryTypeIndex = findDeviceMemoryType(memoryReqs, vk::MemoryPropertyFlagBits::eDeviceLocal);

		vk::MemoryAllocateInfo memAllocInfo(memoryReqs.size(), memoryTypeIndex);
		if ((result = vk::allocateMemory(vDevice, &memAllocInfo, nullptr, &depthStencilMemory)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::allocateMemory -> " + std::to_string((int)result));
		}

		if ((result = vk::bindImageMemory(vDevice, depthStencilImage, depthStencilMemory, 0)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::bindImageMemory -> " + std::to_string((int)result));
		}

		vk::ImageViewCreateInfo imgViewCreateInfo(0, depthStencilImage, vk::ImageViewType::e2D,
			depthStencilFormat, vk::ComponentMapping(/*identity map*/),
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1));
		if ((result = vk::createImageView(vDevice, &imgViewCreateInfo, nullptr, &depthStencilView)) != vk::Result::eVkSuccess)
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
		vk::SubpassDescription subpassDesc(0, vk::PipelineBindPoint::eGraphics,
			0, nullptr, 1, &surfReference, nullptr, &depthReference, 0, nullptr);
		vk::RenderPassCreateInfo renderPassCreateInfo(0, 2, attachDescs, 1, &subpassDesc, 0, nullptr);
		if ((result = vk::createRenderPass(vDevice, &renderPassCreateInfo, nullptr, &renderPass)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::createRenderPass -> " + std::to_string((int)result));
		}
	}

	// pipeline cache setup
	vk::PipelineCache pipelineCache;
	{
		vk::PipelineCacheCreateInfo pipelineCreateInfo(0, 0, nullptr);
		if ((result = vk::createPipelineCache(vDevice, &pipelineCreateInfo, nullptr, &pipelineCache)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::createPipelineCache -> " + std::to_string((int)result));
		}
	}

	// framebuffers setup
	std::vector<vk::Framebuffer> framebuffers(swapChainViews.size());
	{
		vk::ImageView attachments[2]; // color surface, depthstencil pair
		vk::FramebufferCreateInfo fbCreateInfo(0, renderPass, 2, attachments, displayWidth, displayHeight, 1);
		for (size_t i = 0; i < swapChainViews.size(); i++)
		{
			attachments[0] = swapChainViews[i].view;
			attachments[1] = depthStencilView;
			if ((result = vk::createFramebuffer(vDevice, &fbCreateInfo, nullptr, &framebuffers[i])) != vk::Result::eVkSuccess)
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
		if ((result = vk::allocateCommandBuffers(vDevice, &cbufAllocInfo, &layoutCmdBuf)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::allocateCommandBuffers -> " + std::to_string((int)result));
		}

		vk::CommandBufferBeginInfo cbufBegin;
		if ((result = vk::beginCommandBuffer(layoutCmdBuf, &cbufBegin)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::beginCommandBuffer -> " + std::to_string((int)result));
		}

		// transition swapchain images into presentation layout (VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
		for (const auto& swap : swapChainViews)
		{
			vk::ImageMemoryBarrier barrier(
				vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite,
				vk::AccessFlags(),
				vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKhr,
				VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, swap.image,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
				);
			vk::cmdPipelineBarrier(layoutCmdBuf,
				vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe,
				vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &barrier);
		}
		// transition depth stencil image (to VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		vk::ImageMemoryBarrier barrier(
			vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite,
			vk::AccessFlagBits::eDepthStencilAttachmentWrite,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, depthStencilImage,
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1)
			);
		vk::cmdPipelineBarrier(layoutCmdBuf,
			vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe,
			vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &barrier);

		if ((result = vk::endCommandBuffer(layoutCmdBuf)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::endCommandBuffer -> " + std::to_string((int)result));
		}

		// submit
		vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &layoutCmdBuf, 0, nullptr);
		if ((result = vk::queueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::queueSubmit -> " + std::to_string((int)result));
		}
		if ((result = vk::queueWaitIdle(graphicsQueue)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::queueWaitIdle -> " + std::to_string((int)result));
		}

		vk::freeCommandBuffers(vDevice, commandPool, 1, &layoutCmdBuf);
	}

	// setup commandbuffers for swapchain images, plus one extra for post-presentation barrier
	std::vector<vk::CommandBuffer> swapchainCmdBuffers(swapChainViews.size() + 1);
	vk::CommandBuffer presentBarrierCmdBuffer;
	{
		vk::CommandBufferAllocateInfo cmdBufAllocInfo(commandPool, vk::CommandBufferLevel::ePrimary, (uint32_t)swapchainCmdBuffers.size());
		if ((result = vk::allocateCommandBuffers(vDevice, &cmdBufAllocInfo, swapchainCmdBuffers.data())) != vk::Result::eVkSuccess)
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
			if ((result = vk::createBuffer(vDevice, &bufCreateInfo, nullptr, &vertexBuffer)) != vk::Result::eVkSuccess)
			{
				throw std::runtime_error("vk::createBuffer -> " + std::to_string((int)result));
			}
			vk::MemoryRequirements memoryReqs;
			vk::getBufferMemoryRequirements(vDevice, vertexBuffer, &memoryReqs);

			// find memory fitting memoryReqs and host visible
			uint32_t memoryTypeIndex = findDeviceMemoryType(memoryReqs, vk::MemoryPropertyFlagBits::eHostVisible);

			vk::DeviceMemory vBufMemory;
			vk::MemoryAllocateInfo memAllocInfo(memoryReqs.size(), memoryTypeIndex);
			if ((result = vk::allocateMemory(vDevice, &memAllocInfo, nullptr, &vBufMemory)) != vk::Result::eVkSuccess)
			{
				throw std::runtime_error("vk::allocateMemory -> " + std::to_string((int)result));
			}
			void* vBufRamData;
			if ((result = vk::mapMemory(vDevice, vBufMemory, 0, memAllocInfo.allocationSize(), 0, &vBufRamData)) != vk::Result::eVkSuccess)
			{
				throw std::runtime_error("vk::mapMemory -> " + std::to_string((int)result));
			}
			memcpy(vBufRamData, vertexData.data(), vertexBytes);
			vk::unmapMemory(vDevice, vBufMemory);
			if ((result = vk::bindBufferMemory(vDevice, vertexBuffer, vBufMemory, 0)) != vk::Result::eVkSuccess)
			{
				throw std::runtime_error("vk::bindBufferMemory -> " + std::to_string((int)result));
			}
		}
		// index
		{
			uint64_t indexBytes = indexData.size() * sizeof(decltype(indexData)::value_type);

			vk::BufferCreateInfo bufCreateInfo(vk::BufferCreateFlags(), indexBytes, vk::BufferUsageFlagBits::eIndexBuffer, vk::SharingMode::eExclusive, 0, nullptr);
			if ((result = vk::createBuffer(vDevice, &bufCreateInfo, nullptr, &indexBuffer)) != vk::Result::eVkSuccess)
			{
				throw std::runtime_error("vk::createBuffer -> " + std::to_string((int)result));
			}
			vk::MemoryRequirements memoryReqs;
			vk::getBufferMemoryRequirements(vDevice, indexBuffer, &memoryReqs);

			// find memory fitting memoryReqs and host visible
			uint32_t memoryTypeIndex = findDeviceMemoryType(memoryReqs, vk::MemoryPropertyFlagBits::eHostVisible);

			vk::DeviceMemory indexMemory;
			vk::MemoryAllocateInfo memAllocInfo(memoryReqs.size(), memoryTypeIndex);
			if ((result = vk::allocateMemory(vDevice, &memAllocInfo, nullptr, &indexMemory)) != vk::Result::eVkSuccess)
			{
				throw std::runtime_error("vk::allocateMemory -> " + std::to_string((int)result));
			}
			void* indexRamData;
			if ((result = vk::mapMemory(vDevice, indexMemory, 0, memAllocInfo.allocationSize(), 0, &indexRamData)) != vk::Result::eVkSuccess)
			{
				throw std::runtime_error("vk::mapMemory -> " + std::to_string((int)result));
			}
			memcpy(indexRamData, indexData.data(), indexBytes);
			vk::unmapMemory(vDevice, indexMemory);
			if ((result = vk::bindBufferMemory(vDevice, indexBuffer, indexMemory, 0)) != vk::Result::eVkSuccess)
			{
				throw std::runtime_error("vk::bindBufferMemory -> " + std::to_string((int)result));
			}
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
		if ((result = vk::createBuffer(vDevice, &bufCreateInfo, nullptr, &uniformBuffer)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::createBuffer -> " + std::to_string((int)result));
		}
		vk::MemoryRequirements memoryReqs;
		vk::getBufferMemoryRequirements(vDevice, uniformBuffer, &memoryReqs);

		// find memory fitting memoryReqs and host visible
		uint32_t memoryTypeIndex = findDeviceMemoryType(memoryReqs, vk::MemoryPropertyFlagBits::eHostVisible);

		vk::MemoryAllocateInfo memAllocInfo(memoryReqs.size(), memoryTypeIndex);
		if ((result = vk::allocateMemory(vDevice, &memAllocInfo, nullptr, &uniformMemory)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::allocateMemory -> " + std::to_string((int)result));
		}

		// bind memory to buffer
		if ((result = vk::bindBufferMemory(vDevice, uniformBuffer, uniformMemory, 0)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::bindBufferMemory -> " + std::to_string((int)result));
		}

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
		uint8_t *pData;
		if ((result = vk::mapMemory(vDevice, uniformMemory, 0, sizeof(uboInstance), 0, (void**)&pData)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::mapMemory -> " + std::to_string((int)result));
		}
		memcpy(pData, &uboInstance, sizeof(uboInstance));
		vk::unmapMemory(vDevice, uniformMemory);
	}

	// descriptor sets
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::PipelineLayout pipelineLayout;
	{
		vk::DescriptorSetLayoutBinding layoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);
		vk::DescriptorSetLayoutCreateInfo layoutCreateInfo(0, 1, &layoutBinding);
		if ((result = vk::createDescriptorSetLayout(vDevice, &layoutCreateInfo, nullptr, &descriptorSetLayout)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::createDescriptorSetLayout -> " + std::to_string((int)result));
		}

		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(0, 1, &descriptorSetLayout, 0, nullptr);
		if ((result = vk::createPipelineLayout(vDevice, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout)) != vk::Result::eVkSuccess)
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
			vk::ShaderModuleCreateInfo shaderCreateInfo(0, shader.size(), (uint32_t*)shader.data());
			vk::ShaderModule shaderModule;
			vk::Result result = vk::createShaderModule(device, &shaderCreateInfo, nullptr, &shaderModule);
			if (result != vk::Result::eVkSuccess)
			{
				throw std::runtime_error("vk::createShaderModule -> " + std::to_string((int)result));
			}
			return shaderModule;
		};

		std::ifstream vertFile("triangle.vert.spv");
		std::string vertStr((std::istreambuf_iterator<char>(vertFile)), std::istreambuf_iterator<char>());
		std::ifstream fragFile("triangle.frag.spv");
		std::string fragStr((std::istreambuf_iterator<char>(fragFile)), std::istreambuf_iterator<char>());

		vk::PipelineShaderStageCreateInfo shaderStages[2] = {
			vk::PipelineShaderStageCreateInfo(0, vk::ShaderStageFlagBits::eVertex,
				loadShader(vertStr, vDevice),
				"VertexShader", nullptr),
			vk::PipelineShaderStageCreateInfo(0, vk::ShaderStageFlagBits::eFragment,
				loadShader(fragStr, vDevice),
				"FragmentShader", nullptr),
		};

		vk::VertexInputBindingDescription vertexBindDesc(sVertexBufferBindId,
			6 * sizeof(float), // 6 float stride
			vk::VertexInputRate::eVertex);
		std::vector<vk::VertexInputAttributeDescription> vertexAttribDescs = {
			// position attribute (x,y,z)
			vk::VertexInputAttributeDescription(0, sVertexBufferBindId, vk::Format::eR32G32B32Sfloat, 0),
			// color attribute (r,g,b)
			vk::VertexInputAttributeDescription(0, sVertexBufferBindId, vk::Format::eR32G32B32Sfloat, 3 * sizeof(float)),
		};
		vk::PipelineVertexInputStateCreateInfo vertexInputCreate(0, 1, &vertexBindDesc, 
			(uint32_t)vertexAttribDescs.size(), vertexAttribDescs.data());

		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyStateCreate(0, vk::PrimitiveTopology::eTriangleList, false);

		// pViewports and pScissors is dynamic, therefore null here
		vk::PipelineViewportStateCreateInfo viewportStateCreate(0, 1, nullptr, 1, nullptr);

		vk::PipelineRasterizationStateCreateInfo rasterizationStateCreate(0, false, false,
			vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise,
			false, 0, 0, 0, 0);

		vk::PipelineMultisampleStateCreateInfo multisampleStateCreate(0, vk::SampleCountFlagBits::e1, false, 0, nullptr, false, false);

		vk::PipelineDepthStencilStateCreateInfo depthStencilStateCreate(0, true, true, vk::CompareOp::eAlways, false, false,
			vk::StencilOpState(), vk::StencilOpState(), 0, 0);

		vk::PipelineColorBlendAttachmentState blendAttachment;
		vk::PipelineColorBlendStateCreateInfo colorBlendStateCreate(0, false, vk::LogicOp::eClear, 1, &blendAttachment, { 0 });

		std::vector<vk::DynamicState> dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
		vk::PipelineDynamicStateCreateInfo dynamicStateCreate(0, (uint32_t)dynamicStates.size(), dynamicStates.data());

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

		if ((result = vk::createGraphicsPipelines(vDevice, pipelineCache, 1, &graphicsPipelineCreateInfo, nullptr, &pipeline)) != vk::Result::eVkSuccess)
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
		if ((result = vk::createDescriptorPool(vDevice, &descPoolCreate, nullptr, &descriptorPool)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::createDescriptorPool -> " + std::to_string((int)result));
		}
	}

	// descriptor set
	vk::DescriptorSet descriptorSet;
	{
		vk::DescriptorSetAllocateInfo descAllocInfo(descriptorPool, 1, &descriptorSetLayout);
		if ((result = vk::allocateDescriptorSets(vDevice, &descAllocInfo, &descriptorSet)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::allocateDescriptorSets -> " + std::to_string((int)result));
		}

		vk::DescriptorBufferInfo descBufferInfo(uniformBuffer, 0, sizeof(uboInstance));
		vk::WriteDescriptorSet writeDescriptorSet(descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &descBufferInfo, nullptr);
		vk::updateDescriptorSets(vDevice, 1, &writeDescriptorSet, 0, nullptr);
	}

	// swapchain command buffers, load with HelloTriangle
	{
		for (uint32_t i = 0; i < swapchainCmdBuffers.size(); i++)
		{
			auto& framebuffer = framebuffers[i];
			auto& commandBuf = swapchainCmdBuffers[i];
			auto& swapchainImage = swapChainViews[i].image;

			vk::CommandBufferBeginInfo cbufBegin;
			if ((result = vk::beginCommandBuffer(commandBuf, &cbufBegin)) != vk::Result::eVkSuccess)
			{
				throw std::runtime_error("vk::beginCommandBuffer -> " + std::to_string((int)result));
			}

			vk::ClearValue clearValues[] = {
				vk::ClearValue(vk::ClearColorValue(std::array<float, 4>( { 1.f, 0.f, 0.f, 1.f } ))),
				vk::ClearValue(vk::ClearColorValue(std::array<float, 4>( { 1.f, 0.f, 0.f, 0.f } ))),
			};
			vk::RenderPassBeginInfo renderPassBegin(renderPass, framebuffer,
				vk::Rect2D(vk::Offset2D(), vk::Extent2D(displayWidth, displayHeight)),
				2, clearValues);
			vk::cmdBeginRenderPass(commandBuf, &renderPassBegin, vk::SubpassContents::eInline);

			vk::Viewport viewport(0, 0, (float)displayWidth, (float)displayHeight, 0.f, 1.f);
			vk::cmdSetViewport(commandBuf, 0, 1, &viewport);

			vk::Rect2D scissor(vk::Offset2D(), vk::Extent2D(displayWidth, displayHeight));
			vk::cmdSetScissor(commandBuf, 0, 1, &scissor);

			vk::cmdBindDescriptorSets(commandBuf, vk::PipelineBindPoint::eGraphics,
				pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

			vk::cmdBindPipeline(commandBuf, vk::PipelineBindPoint::eGraphics, pipeline);

			vk::cmdBindVertexBuffers(commandBuf, 0, 1, &vertexBuffer, { 0 });
			vk::cmdBindIndexBuffer(commandBuf, indexBuffer, 0, vk::IndexType::eUint32);
			vk::cmdDrawIndexed(commandBuf, 3, 1, 0, 0, 1); // draw 3 verts out of index buffer

			vk::cmdEndRenderPass(commandBuf);

			vk::ImageMemoryBarrier prePresentBarrier(vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlags(),
				vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKhr,
				VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, swapchainImage,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
			vk::cmdPipelineBarrier(commandBuf, vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTopOfPipe,
				vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &prePresentBarrier);

			if ((result = vk::endCommandBuffer(commandBuf)) != vk::Result::eVkSuccess)
			{
				throw std::runtime_error("vk::endCommandBuffer -> " + std::to_string((int)result));
			}
		}
	}

	// loop
	uint32_t swapchainBufferIndex;
	while (true)
	{
		//glfw
		
		vkDeviceWaitIdle(vDevice);

		vk::Semaphore presentationSem;
		if ((result = vk::createSemaphore(vDevice, &vk::SemaphoreCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT), nullptr, &presentationSem)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::createSemaphore -> " + std::to_string((int)result));
		}

		if ((result = vk::acquireNextImageKHR(vDevice, swapchain, UINT64_MAX, presentationSem, vk::Fence(), &swapchainBufferIndex)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::acquireNextImageKHR -> " + std::to_string((int)result));
		}
		vk::CommandBuffer cbuf = swapchainCmdBuffers[swapchainBufferIndex];
		vk::Image swapImage = swapChainViews[swapchainBufferIndex].image;

		vk::SubmitInfo submitInfo(1, &presentationSem, nullptr, 1, &cbuf, 0, nullptr);
		result = vk::queueSubmit(graphicsQueue, 1,
			&vk::SubmitInfo(1, &presentationSem, nullptr, 1, &cbuf, 0, nullptr),
			vk::Fence());
		if (result != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::queueSubmit -> " + std::to_string((int)result));
		}
		
		vk::PresentInfoKHR presentInfo(0, nullptr, 1, &swapchain, &swapchainBufferIndex, nullptr);
		if ((result = vk::queuePresentKHR(graphicsQueue, &presentInfo)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::queuePresentKHR -> " + std::to_string((int)result));
		}
		
		vk::destroySemaphore(vDevice, presentationSem, nullptr);

		if ((result = vk::beginCommandBuffer(presentBarrierCmdBuffer, &vk::CommandBufferBeginInfo())) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::beginCommandBuffer -> " + std::to_string((int)result));
		}

		vk::ImageMemoryBarrier presentBarrier(vk::AccessFlags(), vk::AccessFlagBits::eColorAttachmentWrite,
			vk::ImageLayout::ePresentSrcKhr, vk::ImageLayout::eColorAttachmentOptimal,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, swapImage,
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
		vk::cmdPipelineBarrier(presentBarrierCmdBuffer, vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTopOfPipe,
			vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &presentBarrier);
		
		if ((result = vk::endCommandBuffer(presentBarrierCmdBuffer)) != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::endCommandBuffer -> " + std::to_string((int)result));
		}

		result = vk::queueSubmit(graphicsQueue, 1,
			&vk::SubmitInfo(0, nullptr, nullptr, 1, &presentBarrierCmdBuffer, 0, nullptr),
			vk::Fence());
		if (result != vk::Result::eVkSuccess)
		{
			throw std::runtime_error("vk::queueSubmit -> " + std::to_string((int)result));
		}
	}

	return 0;
}
