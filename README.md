# Baboons-AGMM

Baboons-AGMM is a C++ implementation of the Adaptive Gaussian Mixture Model (AGMM) algorithm for the detection of anomalies in time series data.

## Requirements

- CMake (version 3.0 or higher)
- OpenCV
- (Optional) OpenMP
- (Optional) glfw3 and OpenGL for imGUI

## Building the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/UCSD-E4E/baboon-agmm
    cd baboon-agmm
    ```

2. Create a build directory and change to that directory:
    ```bash
    mkdir build
    cd build
    ```

3. Run CMake to configure the build. To include optional components, add the corresponding flags:
    ```bash
    cmake .. -DWITH_OPENMP=ON -DWITH_IMGUI=ON
    ```

4. Build the project:
    ```bash
    make
    ```

## Running the Project

After building the project, you can run the program with a path to a video file as an argument:

```bash
./bin/Main <video_path> [-r|--record] [-d|--debug] [-s|--disable-shadow]