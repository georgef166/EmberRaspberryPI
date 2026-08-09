cmake_minimum_required(VERSION 3.4)

project(cpp_example)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Reference Arducam demos (kept for context)
add_executable(preview_depth preview_depth.cpp)
target_link_libraries(preview_depth ${CXXLIBS} ${OpenCV_LIBS})

add_executable(capture_raw capture_raw.cpp)
target_link_libraries(capture_raw ${CXXLIBS} ${OpenCV_LIBS})

# Google Technology: TensorFlow Lite (libtensorflow-lite) — optional.
# Powers the on-device MobileNet SSD person detector. There is no apt package
# for the C++ library on Raspberry Pi OS (it must be built from source), so this
# is intentionally NOT marked REQUIRED — the ToF, thermal, and classical CV
# detectors all build and run without it.
set(EMBER_LOCAL_TFLITE_BUILD "${CMAKE_SOURCE_DIR}/.ember-deps/tflite-coral-build")
set(EMBER_LOCAL_TFLITE_SOURCE "${CMAKE_SOURCE_DIR}/.ember-deps/tensorflow-tflite-coral")
set(EMBER_LOCAL_TFLITE_DEP_INCLUDES
    "${EMBER_LOCAL_TFLITE_BUILD}/flatbuffers/include"
    "${EMBER_LOCAL_TFLITE_BUILD}/abseil-cpp"
    "${EMBER_LOCAL_TFLITE_BUILD}/eigen"
    "${EMBER_LOCAL_TFLITE_BUILD}/gemmlowp"
    "${EMBER_LOCAL_TFLITE_BUILD}/ruy"
    "${EMBER_LOCAL_TFLITE_BUILD}/neon2sse")
find_library(TFLITE_LIB tensorflow-lite
    HINTS "${EMBER_LOCAL_TFLITE_BUILD}"
    NO_DEFAULT_PATH)
if(NOT TFLITE_LIB)
    find_library(TFLITE_LIB tensorflow-lite)
endif()
find_path(TFLITE_INCLUDE_DIR tensorflow/lite/interpreter.h
    HINTS "${EMBER_LOCAL_TFLITE_SOURCE}"
    NO_DEFAULT_PATH)
if(NOT TFLITE_INCLUDE_DIR)
    find_path(TFLITE_INCLUDE_DIR tensorflow/lite/interpreter.h)
endif()

# Google Coral Edge TPU (libedgetpu) — optional, and requires TFLite.
find_library(EDGETPU_LIB edgetpu)

# MLX90640 thermal camera (libMLX90640) — optional.
# When present, the binary gains the thermal overlay, fire warning, and
# --detector-source thermal warm-body detector. Build succeeds without it.
find_library(MLX90640_LIB MLX90640)
find_path(MLX90640_INCLUDE_DIR MLX90640_API.h
    PATH_SUFFIXES MLX90640 mlx90640 headers)

# Ember — Tactical Rescue heads-up display
add_executable(
    tactical_rescue
    tactical_rescue.cpp
    tactical_rescue_capture.cpp
    tactical_rescue_ground.cpp   # Ground plane RANSAC + AR navigation grid projection
    tactical_rescue_perception.cpp
    tactical_rescue_render.cpp
    tactical_rescue_stream.cpp  # Browser-viewable MJPEG HUD stream
    tactical_rescue_tflite.cpp   # Google TFLite inference pipeline
    tactical_rescue_thermal.cpp  # MLX90640 thermal capture + warm-body detector
)
target_include_directories(tactical_rescue PRIVATE /usr/include)
target_link_libraries(tactical_rescue ${CXXLIBS} ${OpenCV_LIBS})

# TensorFlow Lite (optional). The Coral backend lives under it: no TFLite, no Coral.
if(TFLITE_LIB)
    message(STATUS "TensorFlow Lite support: ENABLED (${TFLITE_LIB})")
    set_property(TARGET tactical_rescue PROPERTY CXX_STANDARD 17)
    target_compile_definitions(tactical_rescue PRIVATE EMBER_HAVE_TFLITE=1)
    target_link_libraries(tactical_rescue ${TFLITE_LIB})
    get_filename_component(TFLITE_LIB_DIR "${TFLITE_LIB}" DIRECTORY)
    set_property(TARGET tactical_rescue APPEND PROPERTY BUILD_RPATH "${TFLITE_LIB_DIR}")
    if(TFLITE_INCLUDE_DIR)
        target_include_directories(tactical_rescue PRIVATE ${TFLITE_INCLUDE_DIR})
    endif()
    foreach(TFLITE_DEP_INCLUDE ${EMBER_LOCAL_TFLITE_DEP_INCLUDES})
        if(EXISTS "${TFLITE_DEP_INCLUDE}")
            target_include_directories(tactical_rescue PRIVATE "${TFLITE_DEP_INCLUDE}")
        endif()
    endforeach()

    if(EDGETPU_LIB)
        message(STATUS "Coral Edge TPU support: ENABLED (${EDGETPU_LIB})")
        target_compile_definitions(tactical_rescue PRIVATE EMBER_HAVE_EDGETPU=1)
        target_link_libraries(tactical_rescue ${EDGETPU_LIB})
    else()
        message(STATUS "Coral Edge TPU support: DISABLED (libedgetpu not found)")
    endif()
else()
    message(STATUS "TensorFlow Lite support: DISABLED (libtensorflow-lite not found; "
                   "ToF + thermal + classical CV detectors still build). Coral also requires TFLite.")
endif()

if(MLX90640_LIB)
    message(STATUS "MLX90640 thermal support: ENABLED (${MLX90640_LIB})")
    target_compile_definitions(tactical_rescue PRIVATE EMBER_HAVE_MLX90640=1)
    target_link_libraries(tactical_rescue ${MLX90640_LIB})
    if(MLX90640_INCLUDE_DIR)
        target_include_directories(tactical_rescue PRIVATE ${MLX90640_INCLUDE_DIR})
    endif()
else()
    message(STATUS "MLX90640 thermal support: DISABLED (libMLX90640 not found; thermal overlay/detector off)")
endif()
