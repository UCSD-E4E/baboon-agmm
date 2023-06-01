cmake_minimum_required(VERSION 3.0)

project(BackgroundSubtraction)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
set(OpenGL_GL_PREFERENCE "GLVND")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set (CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")
set (CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra")

# Find packages
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)   # Find OpenGL

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/imGUI
    ${OPENGL_INCLUDE_DIR}    # Include OpenGL
)

# Define executable
set(SOURCE_FILES
    BackgroundSubtraction.cpp
    include/AGMM.h
    include/Mixture.h
    include/Gaussian.h
    src/AGMM.cpp
    src/Mixture.cpp
    imGUI/imconfig.h
    imGUI/imgui_demo.cpp
    imGUI/imgui_draw.cpp
    imGUI/imgui_impl_glfw.cpp
    imGUI/imgui_impl_glfw.h
    imGUI/imgui_impl_opengl3_loader.h
    imGUI/imgui_impl_opengl3.cpp
    imGUI/imgui_impl_opengl3.h
    imGUI/imgui_internal.h
    imGUI/imgui_tables.cpp
    imGUI/imgui_widgets.cpp
    imGUI/imgui.cpp
    imGUI/imgui.h
    imGUI/implot_internal.h
    imGUI/implot_items.cpp
    imGUI/implot.cpp
    imGUI/implot.h
    imGUI/imstb_rectpack.h
    imGUI/imstb_textedit.h
    imGUI/imstb_truetype.h
)

add_executable(BackgroundSubtraction ${SOURCE_FILES})

target_include_directories(BackgroundSubtraction PUBLIC src)

# Link libraries
target_link_libraries(BackgroundSubtraction PUBLIC ${OpenCV_LIBS})
target_link_libraries(BackgroundSubtraction PUBLIC glfw)
if(OpenMP_CXX_FOUND)
    target_link_libraries(BackgroundSubtraction PUBLIC OpenMP::OpenMP_CXX)
endif()

# Link OpenGL
target_link_libraries(BackgroundSubtraction PUBLIC OpenGL::GL)