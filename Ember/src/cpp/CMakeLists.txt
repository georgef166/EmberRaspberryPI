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

if(EDGETPU_LIB)
    message(STATUS "Coral Edge TPU support: ENABLED (${EDGETPU_LIB})")
    target_compile_definitions(tactical_rescue PRIVATE EMBER_HAVE_EDGETPU=1)
    target_link_libraries(tactical_rescue ${EDGETPU_LIB})
else()
    message(STATUS "Coral Edge TPU support: DISABLED (libedgetpu not found; build is CPU-only)")
endif()
