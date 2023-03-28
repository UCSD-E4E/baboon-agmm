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
        this->background = frame;
        GaussianBlur(frame, frame, Size(3, 3), 2, 2);

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
        Mixture mixture(this->BM_numberOfGaussians, this->BM_alpha, this->BM_upperboundVariance, this->BM_lowerboundVariance);
        mixture.initializeMixture(initializationData[i]);
        this->mixtures.push_back(mixture);
    }


}

tuple<Mat, Mat, Mat> AGMM::processNextFrame() {
    this->cap >> this->frame;
    this->mask = Mat::zeros(this->rows, this->cols, CV_8U);
    this->result = Mat::zeros(this->rows, this->cols, CV_8UC3);

    // If no more frames, error and deconstruct AGMM
    if (this->frame.empty()) {
        cout << "Error: No more frames in video." << endl;
        this->~AGMM();
    }

    this->backgroundMaintenance();
    bitwise_and(this->frame, this->frame, this->result, this->mask);

    return make_tuple(this->mask, this->result, this->frame);
}



void AGMM::backgroundMaintenance() {
    Mat workingFrame;
    GaussianBlur(this->frame, workingFrame, Size(3, 3), 2, 2);
    Mat foregroundMask = Mat::zeros(this->rows, this->cols, CV_8U);
    

    // Update each mixture and create foreground mask
    for (unsigned int i = 0; i < this->rows; i++) {
        for (unsigned int j = 0; j < this->cols; j++) {
            Vec3b pixel = workingFrame.at<Vec3b>(i, j);
            bool isBackround = this->mixtures[i * this->cols + j].updateMixture(pixel, this->BM_backgroundRatio);
            if (!isBackround) {
                foregroundMask.at<uchar>(i, j) = 255;
            } else {
                this->background.at<Vec3b>(i, j) = pixel;
            }
        }
    }   

    foregroundMask = this->maskCleaner(foregroundMask);

    this->mask = foregroundMask;

}

void AGMM::shadowDetection() {
    Mat hsvFrame, hsvBackground, shadowMask;
    cvtColor(this->frame, hsvFrame, COLOR_BGR2HSV);
    cvtColor(this->background, hsvBackground, COLOR_BGR2HSV);
    shadowMask = Mat::zeros(this->rows, this->cols, CV_8U);
    
    for (unsigned int i = 0; i < this->rows; i++) {
        for (unsigned int j = 0; j < this->cols; j++) {
            double valueRatio = (double)hsvFrame.at<Vec3b>(i, j)[2] / (double)hsvBackground.at<Vec3b>(i, j)[2];
            if (valueRatio > this->SD_valueLowerbound && valueRatio < this->SD_valueUpperbound) {
                int hueDifferenceSum = 0;
                int saturationDifferenceSum = 0;
                int windowArea = 0;
                int minY = max(i - 1, static_cast<unsigned int>(0));
                int maxY = min(i + 1, this->rows - 1);
                int minX = max(j - 1, static_cast<unsigned int>(0));
                int maxX = min(j + 1, this->cols - 1);

                for (int k = minY; k <= maxY; k++) {
                    for (int l = minX; l <= maxX; l++) {
                        int hueDifference = abs(hsvFrame.at<Vec3b>(k, l)[0] - hsvBackground.at<Vec3b>(k, l)[0]);
                        if (hueDifference > 90) {
                            hueDifference = 180 - hueDifference;
                        }
                        hueDifferenceSum += hueDifference;

                        int saturationDifference = abs(hsvFrame.at<Vec3b>(k, l)[1] - hsvBackground.at<Vec3b>(k, l)[1]);
                        saturationDifferenceSum += saturationDifference;
                        windowArea++;
                    }
                }

                if (hueDifferenceSum / windowArea < this->SD_hueThreshold && saturationDifferenceSum / windowArea < this->SD_saturationThreshold) {
                    shadowMask.at<uchar>(i, j) = 255;
                }
            }
        }
    }

    Mat element = getStructuringElement(MORPH_RECT, Size(2 * 2 + 1, 2 * 2 + 1), Point(2, 2));
    Mat cannyFrame, grayFrame, roiMask;

    cvtColor(this->frame, grayFrame, COLOR_BGR2GRAY);

    Canny(grayFrame, cannyFrame, 50, 150);
    threshold(cannyFrame, cannyFrame, 100, 255, THRESH_BINARY);

    morphologyEx(this->mask, roiMask, MORPH_DILATE, element);
    morphologyEx(roiMask, roiMask, MORPH_DILATE, element);

    bitwise_and(roiMask, cannyFrame, roiMask);

    morphologyEx(roiMask, roiMask, MORPH_ERODE, element);

    shadowMask = shadowMask - roiMask;

    this->mask = this->mask - shadowMask;

    this->mask = this->maskCleaner(this->mask);
}

Mat AGMM::maskCleaner(Mat mask) {
    Rect boundingBox;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    Mat cleanedMask = Mat::zeros(this->rows, this->cols, CV_8U);

    for (unsigned int i = 0; i < contours.size(); i++) {
        drawContours(cleanedMask, contours, i, Scalar(255), FILLED, 8, hierarchy, 0);
    }

    return cleanedMask;
}

