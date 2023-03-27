# Baboons-AGMM

Baboons-AGMM is a C++ implementation of the Adaptive Gaussian Mixture Model (AGMM) algorithm for the detection of anomalies in time series data.

## Installation

Requires C++11 or higher and the following libraries:

OpenCV (with contrib modules optional)

## Usage

```bash
# Clone the repository
git clone https://github.com/UCSD-E4E/baboon-agmm/

# Build the project
cd baboon-agmm

cmake .

make

# Run the program
./bin/BackgroundSubtraction <video_path> [-s|--step]
```