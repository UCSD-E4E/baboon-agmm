cmake_minimum_required(VERSION 3.0)

# Define project and version
project(BackgroundSubtraction VERSION 1.0.0)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
set(OpenGL_GL_PREFERENCE "GLVND")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set (CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")
set (CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra")

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)

# Option to include OpenMP
option(WITH_OPENMP "Compile with OpenMP" OFF)

if(WITH_OPENMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DWITH_OPENMP)
endif()

# Option to include imGUI and its dependencies
option(WITH_IMGUI "Compile with imGUI" OFF)

if(WITH_IMGUI)
    find_package(glfw3 REQUIRED)
    find_package(OpenGL REQUIRED)   # Find OpenGL
    add_definitions(-DWITH_IMGUI)
endif()

# Define executable
set(SOURCE_FILES
    src/main.cpp
    src/AGMM.cpp
    src/Mixture.cpp
)

set(HEADER_FILES
    include/AGMM.h
    include/Mixture.h
    include/Gaussian.h
)

if(WITH_IMGUI)
    set(IMGUI_FILES
        third_party/imGUI/imconfig.h
        third_party/imGUI/imgui_demo.cpp
        third_party/imGUI/imgui_draw.cpp
        third_party/imGUI/imgui_impl_glfw.cpp
        third_party/imGUI/imgui_impl_glfw.h
        third_party/imGUI/imgui_impl_opengl3_loader.h
        third_party/imGUI/imgui_impl_opengl3.cpp
        third_party/imGUI/imgui_impl_opengl3.h
        third_party/imGUI/imgui_internal.h
        third_party/imGUI/imgui_tables.cpp
        third_party/imGUI/imgui_widgets.cpp
        third_party/imGUI/imgui.cpp
        third_party/imGUI/imgui.h
        third_party/imGUI/implot_internal.h
        third_party/imGUI/implot_items.cpp
        third_party/imGUI/implot.cpp
        third_party/imGUI/implot.h
        third_party/imGUI/imstb_rectpack.h
        third_party/imGUI/imstb_textedit.h
        third_party/imGUI/imstb_truetype.h
    )
endif()

add_executable(Main ${SOURCE_FILES} ${HEADER_FILES} ${IMGUI_FILES})

# Include directories
target_include_directories(Main PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    include
)

if(WITH_IMGUI)
    target_include_directories(Main PUBLIC
        ${CMAKE_SOURCE_DIR}/third_party/imGUI
        ${OPENGL_INCLUDE_DIR}    # Include OpenGL
    )
endif()

# Link libraries
target_link_libraries(Main PUBLIC ${OpenCV_LIBS})

if(WITH_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(Main PUBLIC OpenMP::OpenMP_CXX)
endif()

if(WITH_IMGUI)
    target_link_libraries(Main PUBLIC glfw)
    target_link_libraries(Main PUBLIC OpenGL::GL)
endif()