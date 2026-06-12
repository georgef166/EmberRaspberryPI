cmake_minimum_required(VERSION 3.4)

project(cpp_example)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Reference Arducam demos (kept for context)
add_executable(preview_depth preview_depth.cpp)
target_link_libraries(preview_depth ${CXXLIBS} ${OpenCV_LIBS})

add_executable(capture_raw capture_raw.cpp)
target_link_libraries(capture_raw ${CXXLIBS} ${OpenCV_LIBS})

# Google Technology: TensorFlow Lite (libtensorflow-lite v2.20)
# Used for on-device MobileNet SSD person detection in tactical_rescue_tflite.cpp
find_library(TFLITE_LIB tensorflow-lite REQUIRED)

# Google Coral Edge TPU (libedgetpu) — optional.
# When present, the tactical_rescue binary gains the --edgetpu backend that
# offloads inference to an attached Coral. The build still succeeds without it
# (CPU-only), so this is intentionally NOT marked REQUIRED.
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
    tactical_rescue_perception.cpp
    tactical_rescue_render.cpp
    tactical_rescue_tflite.cpp   # Google TFLite inference pipeline
    tactical_rescue_thermal.cpp  # MLX90640 thermal capture + warm-body detector
)
target_include_directories(tactical_rescue PRIVATE /usr/include)
# Link TensorFlow Lite (Google) alongside OpenCV and the Arducam SDK
target_link_libraries(tactical_rescue ${CXXLIBS} ${OpenCV_LIBS} ${TFLITE_LIB})

if(EDGETPU_LIB)
    message(STATUS "Coral Edge TPU support: ENABLED (${EDGETPU_LIB})")
    target_compile_definitions(tactical_rescue PRIVATE EMBER_HAVE_EDGETPU=1)
    target_link_libraries(tactical_rescue ${EDGETPU_LIB})
else()
    message(STATUS "Coral Edge TPU support: DISABLED (libedgetpu not found; build is CPU-only)")
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
