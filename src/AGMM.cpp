/**
 * @file AGMM.cpp
 * @brief Class to implement the adaptive Gaussian mixture model (AGMM) algorithm.
 * The algorithm is described in the paper:
 * "Regularized Background Adaptation: A Novel Learning Rate Control Scheme for Gaussian Mixture Modeling"
 * by Horng-Horn Lin, Jen-Hui Chuang, and Tyng-Luh Liu.
 * @author Tyler Flar
*/
#include "../include/AGMM.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

AGMM::AGMM(string videoPath) {
    this->cap = VideoCapture(videoPath);

    // If video cannot be opened, error and deconstruct AGMM
    if (!this->cap.isOpened()) {
        cout << "Error: Video cannot be opened." << endl;
        this->~AGMM();
        return;
    }

    this->rows = cap.get(CAP_PROP_FRAME_HEIGHT);
    this->cols = cap.get(CAP_PROP_FRAME_WIDTH);
    this->numberOfPixels = this->rows * this->cols;
}

AGMM::~AGMM() {
    this->cap.release();
}

void AGMM::initializeModel(int numberOfFrames) {
    // Vector with initialization data for GMM. 
    // Each pixel has a vector of Vec3b values.
    // The size of the individual vectors is equal to the number of frames. 
    vector<vector<Vec3b>> initializationData(this->numberOfPixels);
    Mat frame;
    for (int i = 0; i < numberOfFrames; i++) {
        this->cap >> frame;
        GaussianBlur(frame, frame, Size(9, 9), 2, 2);

        // If no more frames, error and deconstruct AGMM
        if (frame.empty()) {
            cout << "Error: No more frames in video." << endl;
            this->~AGMM();
            return;
        }

        for (unsigned int j = 0; j < this->rows; j++) {
            for (unsigned int k = 0; k < this->cols; k++) {

                initializationData[j * this->cols + k].push_back(frame.at<Vec3b>(j, k));
            }
        }
    }

    for (unsigned int i = 0; i < this->numberOfPixels; i++) {
        Mixture mixture(this->numberOfGaussians, this->alpha, this->upperboundVariance, this->lowerboundVariance);
        mixture.initializeMixture(initializationData[i]);
        this->mixtures.push_back(mixture);
    }
}

tuple<Mat, Mat, Mat> AGMM::processNextFrame() {
    Mat frame;
    this->cap >> frame;
    GaussianBlur(frame, frame, Size(9, 9), 2, 2);

    // If no more frames, error and deconstruct AGMM
    if (frame.empty()) {
        cout << "Error: No more frames in video." << endl;
        this->~AGMM();
    }

    Mat foregroundMask = Mat::zeros(this->rows, this->cols, CV_8U);
    Mat result = Mat::zeros(this->rows, this->cols, CV_8UC3);
    

    // Update each mixture and create foreground mask
    for (unsigned int i = 0; i < this->rows; i++) {
        for (unsigned int j = 0; j < this->cols; j++) {
            Vec3b pixel = frame.at<Vec3b>(i, j);
            bool isBackround = this->mixtures[i * this->cols + j].updateMixture(pixel, this->backgroundRatio);
            if (!isBackround) {
                foregroundMask.at<uchar>(i, j) = 255;
            }
        }
    }

    bitwise_and(frame, frame, result, foregroundMask);

    return make_tuple(foregroundMask, result, frame);
}
