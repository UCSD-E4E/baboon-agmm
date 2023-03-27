cmake_minimum_required(VERSION 3.0)

project(BackgroundSubtraction)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")

set(CMAKE_CXX_FLAGS_DEBUG "-g")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(BackgroundSubtraction BackgroundSubtraction.cpp  include/AGMM.h include/Mixture.h include/Gaussian.h src/AGMM.cpp src/Mixture.cpp src/Gaussian.cpp)

target_include_directories(BackgroundSubtraction PUBLIC src)

target_link_libraries(BackgroundSubtraction ${OpenCV_LIBS})