cmake_minimum_required(VERSION 3.18)
project(NovaPanoramaGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Jetson AGX Orin = Ampere sm_87. Override with -DCMAKE_CUDA_ARCHITECTURES=... if needed.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

find_package(ZED REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${ZED_LIBRARY_DIR})

if(CMAKE_SYSTEM_PROCESSOR MATCHES aarch64)
    add_definitions(-DJETSON_STYLE)
endif()

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/PanoramaViewer.cpp
    src/kernels.cu
)

target_link_libraries(${PROJECT_NAME}
    ${ZED_LIBRARIES}
    CUDA::cudart
    GLEW::GLEW
    GLUT::GLUT
    OpenGL::GL
)
