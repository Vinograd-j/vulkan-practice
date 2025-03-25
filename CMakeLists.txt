cmake_minimum_required(VERSION 3.30)
project(vulkan_practice)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)

include(FetchContent)
FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG master)
FetchContent_Declare(glad GIT_REPOSITORY https://github.com/SasLuca/MyGlad.git GIT_TAG master)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_MakeAvailable(glad)

add_executable(vulkan_practice main.cpp)
target_link_libraries(vulkan_practice glfw Vulkan::Vulkan)
