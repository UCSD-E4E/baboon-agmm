cmake_minimum_required(VERSION 3.0)

project(BackgroundSubtraction)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set (CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")

set (CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(BackgroundSubtraction BackgroundSubtraction.cpp  include/AGMM.h include/Mixture.h include/Gaussian.h src/AGMM.cpp src/Mixture.cpp src/Gaussian.cpp)

target_include_directories(BackgroundSubtraction PUBLIC src)

target_link_libraries(BackgroundSubtraction PUBLIC ${OpenCV_LIBS})