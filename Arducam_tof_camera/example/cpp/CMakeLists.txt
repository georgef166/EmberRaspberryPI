cmake_minimum_required(VERSION 3.4)

project(cpp_example)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(preview_depth preview_depth.cpp)
target_link_libraries(preview_depth ${CXXLIBS} ${OpenCV_LIBS})

add_executable(capture_raw capture_raw.cpp)
target_link_libraries(capture_raw ${CXXLIBS} ${OpenCV_LIBS})

add_executable(preview_usb preview_by_usb.cpp)
target_link_libraries(preview_usb ${CXXLIBS} ${OpenCV_LIBS})

find_library(TFLITE_LIB tensorflow-lite REQUIRED)

add_executable(
    tactical_rescue
    tactical_rescue.cpp
    tactical_rescue_capture.cpp
    tactical_rescue_perception.cpp
    tactical_rescue_render.cpp
    tactical_rescue_tflite.cpp
)
target_include_directories(tactical_rescue PRIVATE /usr/include)
target_link_libraries(tactical_rescue ${CXXLIBS} ${OpenCV_LIBS} ${TFLITE_LIB})

aux_source_directory(example DIR_SRCS)
add_executable(example ${DIR_SRCS})
target_link_libraries(example ${CXXLIBS} ${OpenCV_LIBS})
