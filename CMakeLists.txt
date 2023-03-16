cmake_minimum_required(VERSION 3.0)

project(BackgroundSubtraction)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(BackgroundSubtraction src/BackgroundSubtraction.cpp src/AGMM.cpp src/AGMM.hpp)

target_include_directories(BackgroundSubtraction PUBLIC src)

target_link_libraries(BackgroundSubtraction ${OpenCV_LIBS})