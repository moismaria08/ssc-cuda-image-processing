cmake_minimum_required(VERSION 3.18)
project(cuda_image_processing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

set(CPU_SOURCES
    src/main.cpp
    src/cpu_processing.cpp
    src/utils.cpp
)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "75;87")

    set(CUDA_SOURCES
        cuda/grayscale_kernel.cu
        cuda/blur_kernel.cu
        cuda/blur_shared_kernel.cu
        cuda/sobel_kernel.cu
        cuda/sobel_shared_kernel.cu
        cuda/pipeline.cu
    )

    add_executable(image_pipeline ${CPU_SOURCES} ${CUDA_SOURCES})
    target_compile_definitions(image_pipeline PRIVATE USE_CUDA)
else()
    message(STATUS "CUDA nu a fost detectat. Se compileaza doar varianta CPU.")
    add_executable(image_pipeline ${CPU_SOURCES})
endif()