cmake_minimum_required(VERSION 3.30)
project(vulkan_practice)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup")
endif ()

find_package(Vulkan REQUIRED)

include(FetchContent)
FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG master)
FetchContent_Declare(glad GIT_REPOSITORY https://github.com/SasLuca/MyGlad.git GIT_TAG master)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_MakeAvailable(glad)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_executable(vulkan_practice WIN32 summer_realization.cpp)
else()
    add_executable(vulkan_practice summer_realization.cpp)
endif ()

target_link_libraries(vulkan_practice glad glfw Vulkan::Vulkan)