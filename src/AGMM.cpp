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

AGMM::AGMM(string videoPath)
{
    this->cap = VideoCapture(videoPath);

    // If video cannot be opened, error and deconstruct AGMM
    if (!this->cap.isOpened())
    {
        cout << "Error: Video cannot be opened." << endl;
        this->~AGMM();
        return;
    }

    this->rows = cap.get(CAP_PROP_FRAME_HEIGHT);
    this->cols = cap.get(CAP_PROP_FRAME_WIDTH);
}

AGMM::~AGMM()
{
    this->cap.release();
}

void AGMM::initializeModel()
{
    // Vector with initialzation data for GMM.
    // Each pixel has a vector of Vec3b values.
    // The size of the individual vectors is equal to the number of frames used for initialization.
    vector<vector<Vec3b>> pixels(this->rows * this->cols, vector<Vec3b>());

    // Initialize the model with the first 10 frames
    for (int i = 0; i < 10; i++)
    {
        this->cap >> this->frame;
        this->background = this->frame.clone();

        // If no more frames, error and deconstruct AGMM
        if (this->frame.empty())
        {
            cout << "Error: No more frames in video." << endl;
            this->~AGMM();
            return;
        }

        Mat workingFrame;
        GaussianBlur(this->frame, workingFrame, Size(9, 9), 2, 2);

        for (unsigned int i = 0; i < this->rows; i++)
        {
            for (unsigned int j = 0; j < this->cols; j++)
            {
                Vec3b pixel = workingFrame.at<Vec3b>(i, j);
                pixels[i * this->cols + j].push_back(pixel);
            }
        }
    }

    // Give each mixture a vector of pixels
    for (unsigned int i = 0; i < this->rows; i++)
    {
        for (unsigned int j = 0; j < this->cols; j++)
        {
            Mixture mixture = Mixture(this->BM_numberOfGaussians, this->BM_alpha, this->BM_beta_b, this->BM_beta_s, this->BM_beta_sf, this->BM_beta_mf);
            this->mixtures.push_back(mixture);
            mixtures[i * this->cols + j].initializeMixture(pixels[i * this->cols + j]);
        }
    }

}

tuple<Mat, Mat, Mat> AGMM::processNextFrame()
{
    this->cap >> this->frame;
    this->mask = Mat::zeros(this->rows, this->cols, CV_8U);
    this->result = Mat::zeros(this->rows, this->cols, CV_8UC3);

    // If no more frames, error and deconstruct AGMM
    if (this->frame.empty())
    {
        cout << "Error: No more frames in video." << endl;
    }

    this->backgroundModelMaintenance();
    this->foregroundPixelIdentification();
    //this->shadowDetection();
    //this->objectExtraction();
    bitwise_and(this->frame, this->frame, this->result, this->mask);

    return make_tuple(this->mask, this->result, this->frame);
}

void AGMM::backgroundModelMaintenance()
{
    Mat workingFrame;
    GaussianBlur(this->frame, workingFrame, Size(9, 9), 2, 2);

    for (unsigned int i = 0; i < this->rows; i++) {
        for (unsigned int j = 0; j < this->cols; j++) {
            Vec3b pixel = workingFrame.at<Vec3b>(i, j);
            this->mixtures[i * this->cols + j].updateMixture(pixel);

        }
    }
}

void AGMM::foregroundPixelIdentification() {
    Mat workingFrame;
    GaussianBlur(this->frame, workingFrame, Size(9, 9), 2, 2);
    Mat foregroundMask = Mat::zeros(this->rows, this->cols, CV_8U);

    for (unsigned int i = 0; i < this->rows; i++) {
        for (unsigned int j = 0; j < this->cols; j++) {
            Vec3b pixel = workingFrame.at<Vec3b>(i, j);
            if (this->mixtures[i * this->cols + j].isForegroundPixel(pixel)) {
                foregroundMask.at<uchar>(i, j) = 255;
            } else {
                this->background.at<Vec3b>(i, j) = this->frame.at<Vec3b>(i, j);
            }
        }
    }

    this->mask = foregroundMask;
}

void AGMM::shadowDetection()
{
    Mat hsvFrame, hsvBackground, shadowMask;
    cvtColor(this->frame, hsvFrame, COLOR_BGR2HSV);
    cvtColor(this->background, hsvBackground, COLOR_BGR2HSV);
    shadowMask = Mat::zeros(this->rows, this->cols, CV_8U);

    for (unsigned int i = 0; i < this->rows; i++)
    {
        for (unsigned int j = 0; j < this->cols; j++)
        {
            double valueRatio = (double)hsvFrame.at<Vec3b>(i, j)[2] / (double)hsvBackground.at<Vec3b>(i, j)[2];
            if (valueRatio > this->SD_valueLowerbound && valueRatio < this->SD_valueUpperbound)
            {
                int hueDifferenceSum = 0;
                int saturationDifferenceSum = 0;
                int windowArea = 0;
                int minY = max(i - 1, static_cast<unsigned int>(0));
                int maxY = min(i + 1, this->rows - 1);
                int minX = max(j - 1, static_cast<unsigned int>(0));
                int maxX = min(j + 1, this->cols - 1);

                for (int k = minY; k <= maxY; k++)
                {
                    for (int l = minX; l <= maxX; l++)
                    {
                        int hueDifference = abs(hsvFrame.at<Vec3b>(k, l)[0] - hsvBackground.at<Vec3b>(k, l)[0]);
                        if (hueDifference > 90)
                        {
                            hueDifference = 180 - hueDifference;
                        }
                        hueDifferenceSum += hueDifference;

                        int saturationDifference = abs(hsvFrame.at<Vec3b>(k, l)[1] - hsvBackground.at<Vec3b>(k, l)[1]);
                        saturationDifferenceSum += saturationDifference;
                        windowArea++;
                    }
                }

                if (hueDifferenceSum / windowArea < this->SD_hueThreshold && saturationDifferenceSum / windowArea < this->SD_saturationThreshold)
                {
                    shadowMask.at<uchar>(i, j) = 255;
                }
            }
        }
    }

    // Mat element = getStructuringElement(MORPH_RECT, Size(2 * 2 + 1, 2 * 2 + 1), Point(2, 2));
    // Mat cannyFrame, grayFrame, roiMask;

    // cvtColor(this->frame, grayFrame, COLOR_BGR2GRAY);

    // Canny(grayFrame, cannyFrame, 50, 150);
    // threshold(cannyFrame, cannyFrame, 100, 255, THRESH_BINARY);

    // morphologyEx(this->mask, roiMask, MORPH_DILATE, element);
    // morphologyEx(roiMask, roiMask, MORPH_DILATE, element);

    // bitwise_and(roiMask, cannyFrame, roiMask);

    // morphologyEx(roiMask, roiMask, MORPH_ERODE, element);

    // shadowMask = shadowMask - roiMask;

    this->mask = this->mask - shadowMask;
}

void AGMM::objectExtraction()
{
    Mat element = getStructuringElement(MORPH_RECT, Size(2 * 2 + 1, 2 * 2 + 1), Point(2, 2));
    morphologyEx(this->mask, this->mask, MORPH_CLOSE, element);
    morphologyEx(this->mask, this->mask, MORPH_OPEN, element);

    Rect boundingBox;
    vector<vector<Point>> contours;
    vector<int> contourIndices;
    vector<Vec4i> hierarchy;

    findContours(this->mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    Mat cleanedMask = Mat::zeros(this->rows, this->cols, CV_8U);

    for (unsigned int i = 0; i < contours.size(); i++)
    {
        boundingBox = boundingRect(contours[i]);
        if (boundingBox.width > 4 && boundingBox.height > 4)
        {
            contourIndices.push_back(i);
        }
    }

    for (unsigned int i = 0; i < contourIndices.size(); i++)
    {
        drawContours(cleanedMask, contours, contourIndices[i], Scalar(255), FILLED, 8, hierarchy, 0);
    }

    this->mask = cleanedMask;
}
