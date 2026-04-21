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

# Ember — Tactical Rescue heads-up display
add_executable(
    tactical_rescue
    tactical_rescue.cpp
    tactical_rescue_capture.cpp
    tactical_rescue_perception.cpp
    tactical_rescue_render.cpp
    tactical_rescue_tflite.cpp  # Google TFLite inference pipeline
)
target_include_directories(tactical_rescue PRIVATE /usr/include)
# Link TensorFlow Lite (Google) alongside OpenCV and the Arducam SDK
target_link_libraries(tactical_rescue ${CXXLIBS} ${OpenCV_LIBS} ${TFLITE_LIB})
