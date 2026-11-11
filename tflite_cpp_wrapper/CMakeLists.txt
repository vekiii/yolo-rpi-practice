cmake_minimum_required(VERSION 3.24)
project(yolo_tflite LANGUAGES C CXX)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    set(BUILD_WITH_STATIC_CRT OFF CACHE BOOL
        "Use dynamic MSVC runtime for OpenCV" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Compatibility for dependency CMake files")
set(TFLITE_ENABLE_XNNPACK ON CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_NNAPI OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_EXTERNAL_DELEGATE OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(tensorflow GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git GIT_TAG v2.20.0 GIT_SHALLOW TRUE SOURCE_SUBDIR tensorflow/lite)
FetchContent_MakeAvailable(tensorflow)
target_compile_definitions(tensorflow-lite PUBLIC
    TF_MAJOR_VERSION=2
    TF_MINOR_VERSION=20
    TF_PATCH_VERSION=0
    "TF_VERSION_SUFFIX=\"\""
)
set(BUILD_LIST core,imgproc,imgcodecs CACHE STRING "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
set(WITH_FFMPEG OFF CACHE BOOL "" FORCE)
set(WITH_OPENCL OFF CACHE BOOL "" FORCE)
set(WITH_IPP OFF CACHE BOOL "" FORCE)
set(WITH_ITT OFF CACHE BOOL "" FORCE)
set(WITH_PROTOBUF OFF CACHE BOOL "" FORCE)
set(WITH_WEBP OFF CACHE BOOL "" FORCE)
set(WITH_OPENEXR OFF CACHE BOOL "" FORCE)
set(WITH_AVIF OFF CACHE BOOL "" FORCE)
FetchContent_Declare(opencv GIT_REPOSITORY https://github.com/opencv/opencv.git GIT_TAG 4.11.0 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(opencv)
FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.11.3 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(json)
add_executable(yolo_tflite main.cpp)
target_include_directories(yolo_tflite PRIVATE
    ${opencv_SOURCE_DIR}/modules/core/include
    ${opencv_SOURCE_DIR}/modules/imgproc/include
    ${opencv_SOURCE_DIR}/modules/imgcodecs/include
    ${opencv_BINARY_DIR}
    ${CMAKE_BINARY_DIR}
)
target_link_libraries(yolo_tflite PRIVATE tensorflow-lite opencv_core opencv_imgproc opencv_imgcodecs nlohmann_json::nlohmann_json)

set_target_properties(yolo_tflite PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
