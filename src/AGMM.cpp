/**
 * @file AGMM.cpp
 * @brief Class to implement the adaptive Gaussian mixture model (AGMM)
 * algorithm. The algorithm is described in the paper: "Regularized Background
 * Adaptation: A Novel Learning Rate Control Scheme for Gaussian Mixture
 * Modeling" by Horng-Horn Lin, Jen-Hui Chuang, and Tyng-Luh Liu.
 * @author Tyler Flar
 */
#include "../include/AGMM.h"
#include <opencv2/opencv.hpp>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

const int BlurSize = 3;

const int BM_numberOfGaussians = 100;
const double BM_alpha = 0.025;
const double BM_beta_b = 0.01;
const double BM_beta_d = 1.0 / 100.0;
const double BM_beta_s = 1.0 / 900.0;
const double BM_beta_m = 1.0 / 6000.0;

AGMM::AGMM(std::string videoPath)
{
    this->cap = cv::VideoCapture(videoPath);

    // If video cannot be opened, error and deconstruct AGMM
    if (!this->cap.isOpened())
    {
        std::cout << "Error: Video cannot be opened." << std::endl;
        this->~AGMM();
        return;
    }

    this->rows = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    this->cols = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    this->fps = cap.get(cv::CAP_PROP_FPS);
}

AGMM::AGMM(std::string videoPath, bool debug, bool disableShadow)
{
    this->debug = debug;
    this->disableShadow = disableShadow;
    this->cap = cv::VideoCapture(videoPath);

    // If video cannot be opened, error and deconstruct AGMM
    if (!this->cap.isOpened())
    {
        std::cout << "Error: Video cannot be opened." << std::endl;
        this->~AGMM();
        return;
    }

    this->rows = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    this->cols = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    this->fps = cap.get(cv::CAP_PROP_FPS);
}

AGMM::~AGMM() { this->cap.release(); }

void AGMM::initializeModel()
{
    this->cap >> this->frame;

    cv::Mat workingFrame;
    cvtColor(this->frame, workingFrame, cv::COLOR_BGR2GRAY);
    GaussianBlur(workingFrame, workingFrame, cv::Size(BlurSize, BlurSize), 0);

    // Give each mixture a vector of pixels
    for (unsigned int i = 0; i < this->rows; i++)
    {
        for (unsigned int j = 0; j < this->cols; j++)
        {
            double intensity = static_cast<double>(workingFrame.at<uchar>(i, j));
            Mixture mixture = Mixture(BM_numberOfGaussians, BM_alpha, BM_beta_b,
                                      BM_beta_d, BM_beta_s, BM_beta_m);
            this->mixtures.push_back(mixture);
            mixtures[i * this->cols + j].initializeMixture(intensity);
        }
    }
}

std::vector<cv::Mat> AGMM::processNextFrame()
{
    this->cap >> this->frame;

    this->objectMask = cv::Mat::zeros(this->rows, this->cols, CV_8U);
    this->shadowMask = cv::Mat::zeros(this->rows, this->cols, CV_8U);
    this->finalMask = cv::Mat::zeros(this->rows, this->cols, CV_8U);
    this->result = cv::Mat::zeros(this->rows, this->cols, CV_8UC3);

    // If no more frames, error and deconstruct AGMM
    if (this->frame.empty())
    {
        std::cout << "Error: No more frames in video." << std::endl;
        return std::vector<cv::Mat>();
    }

    this->backgroundModelMaintenance();
    this->foregroundPixelIdentification();
    if (!this->disableShadow)
        this->shadowDetection();
    this->finalMask = this->objectMask - this->shadowMask;
    this->objectExtraction();
    this->objectTypeClassification();
    cv::bitwise_and(this->frame, this->frame, this->result, this->finalMask);

    std::vector<cv::Mat> output;
    output.push_back(this->objectMask);
    output.push_back(this->shadowMask);
    output.push_back(this->finalMask);
    output.push_back(this->frame);

    return output;
}

void AGMM::backgroundModelMaintenance()
{
    cv::Mat workingFrame;
    cvtColor(this->frame, workingFrame, cv::COLOR_BGR2GRAY);
    GaussianBlur(workingFrame, workingFrame, cv::Size(BlurSize, BlurSize), 0);

#ifdef WITH_OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (unsigned int i = 0; i < this->rows; i++)
    {
        for (unsigned int j = 0; j < this->cols; j++)
        {
            double pixel = static_cast<double>(workingFrame.at<uchar>(i, j));
            this->mixtures[i * this->cols + j].updateMixture(pixel);
        }
    }
}

void AGMM::foregroundPixelIdentification()
{
    cv::Mat foregroundMask = cv::Mat::zeros(this->rows, this->cols, CV_8U);

#ifdef WITH_OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (unsigned int i = 0; i < this->rows; i++)
    {
        for (unsigned int j = 0; j < this->cols; j++)
        {
            if (this->mixtures[i * this->cols + j].isForegroundPixel())
                foregroundMask.at<uchar>(i, j) = 255;
        }
    }

    this->objectMask = foregroundMask;
    this->finalMask = foregroundMask;
}

double medianMat(cv::Mat Input)
{
    Input = Input.reshape(0, 1); // spread Input Mat to single row
    std::vector<double> vecFromMat;
    Input.copyTo(vecFromMat); // Copy Input Mat to vector vecFromMat
    nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2,
                vecFromMat.end());
    return vecFromMat[vecFromMat.size() / 2];
}

void AGMM::shadowDetection()
{
    cv::Mat workingFrame;
    cvtColor(this->frame, workingFrame, cv::COLOR_BGR2GRAY);
    GaussianBlur(workingFrame, workingFrame, cv::Size(BlurSize, BlurSize), 0);
    
    // Reference Image
    cv::Mat reference = cv::Mat::zeros(this->rows, this->cols, CV_8U);
    this->shadowMask = reference;
    
    for (unsigned int i = 0; i < this->rows; i++) {
        for (unsigned int j = 0; j < this->cols; j++) {
            std::vector<Gaussian> gaussians = 
                    this->mixtures[i * this->cols + j].getGaussians();
            double average = 0;
            for(unsigned int i = 0; i < 100; i++)
            {
                average += (gaussians[i].getMean() * gaussians[i].getWeight());
            }
            reference.at<double>(i, j) = average;
        }
    }
    

    // Threshold Image (frame differencing operation)
    // L-filters (linear combination of the ordered samples of the image sequence)
    cv::Mat diffImage;
    absdiff(reference, workingFrame, diffImage);

    double median = medianMat(diffImage);
    cv::Mat madImage;
    absdiff(diffImage - median, cv::Mat::zeros(diffImage.size(), diffImage.type()), madImage);
    double mad = medianMat(madImage);
    double threshold = median + 3 * 1.4826 * mad;
    cv::Mat thresholdedImage;
    cv::threshold(diffImage, thresholdedImage, threshold, 255, cv::THRESH_BINARY);

    // Connectivity preserving thresholding
    cv::Mat output = thresholdedImage.clone();
    cv::Mat thresholdedImage8U;
    thresholdedImage.convertTo(thresholdedImage8U, CV_8U);
    cv::Mat labels;
    int num_objects = connectedComponents(thresholdedImage8U, labels, 8, CV_32S);
    
    // Thresholding with hysteresis

    // Region Growing (single pass neighbourhood connectivity algorithm)
    // Gain calculation and checking
    for (unsigned int i = 0; i < this->rows; i++) {
        for (unsigned int j = 0; j < this->cols; j++) {
            double gain = reference.at<double>(i, j) 
                    / workingFrame.at<double>(i, j);
        }
    }

    // Merging regions
    

    // Test
    output.convertTo(output, CV_8UC3);

    // Update the final mask
    this->shadowMask = output;
    




//     cv::Mat background(this->rows, this->cols, CV_64FC1);

// #ifdef WITH_OPENMP
// #pragma omp parallel for collapse(2)
// #endif
//     for (unsigned int i = 0; i < this->rows; i++)
//     {
//         for (unsigned int j = 0; j < this->cols; j++)
//         {
//             std::vector<Gaussian> gaussians =
//                 this->mixtures[i * this->cols + j].getGaussians();

//             // Extract the mean of each Gaussian.
//             std::vector<double> means(gaussians.size());
//             transform(gaussians.begin(), gaussians.end(), means.begin(),
//                       [](const Gaussian &g)
//                       { return g.getMean(); });

//             // Sort the means.
//             sort(means.begin(), means.end());

//             // Compute the median and set it as the background pixel intensity.
//             double median = means[means.size() / 2];
//             background.at<double>(i, j) = median;
//         }
//     }

//     assert(background.size() == this->frame.size());
//     cv::Mat grayFrame;
//     if (this->frame.channels() == 3)
//         cvtColor(this->frame, grayFrame, cv::COLOR_BGR2GRAY);
//     else
//         grayFrame = this->frame.clone();

//     // Convert grayFrame to double precision.
//     grayFrame.convertTo(grayFrame, CV_64FC1);

//     // Compute the absolute difference image
//     cv::Mat diffImage;
//     cv::absdiff(grayFrame, background, diffImage);

//     // Compute the median of the difference image
//     double median = medianMat(diffImage);

//     // Compute the Median Absolute Deviation (MAD)
//     cv::Mat madImage;
//     absdiff(diffImage - median,
//             cv::Mat::zeros(diffImage.size(), diffImage.type()), madImage);
//     double mad = medianMat(madImage);

//     // Compute the threshold
//     double threshold = median + 3 * 1.4826 * mad;

//     // Threshold the difference image
//     cv::Mat thresholdedImage;
//     cv::threshold(diffImage, thresholdedImage, threshold, 255, cv::THRESH_BINARY);

//     // Copy the thresholded image to create an output image
//     cv::Mat output = thresholdedImage.clone();

//     // Convert thresholded image to 8-bit
//     cv::Mat thresholdedImage8U;
//     thresholdedImage.convertTo(thresholdedImage8U, CV_8U);

//     // Connected Components compulation
//     cv::Mat labels;
//     int num_objects = connectedComponents(thresholdedImage8U, labels, 8, CV_32S);

//     // Array to hold the number of pixels in each component
//     std::vector<int> component_sizes(num_objects, 0);

//     // Iterate over the labels to count the number of pixels in each component
//     for (int i = 0; i < labels.rows; i++)
//     {
//         for (int j = 0; j < labels.cols; j++)
//         {
//             component_sizes[labels.at<int>(i, j)]++;
//         }
//     }

//     // Minimum size for the component to be kept
//     const int min_size = 2;

//     // Iterate over the image to update the output image
//     for (int i = 0; i < labels.rows; i++)
//     {
//         for (int j = 0; j < labels.cols; j++)
//         {
//             // If the component is too small, set the pixel to 0
//             if (component_sizes[labels.at<int>(i, j)] < min_size)
//             {
//                 output.at<uchar>(i, j) = 0;
//             }
//         }
//     }

//     // Override the thresholded image with the updated output image
//     thresholdedImage = output;

//     // 2-level hysteresis thresholding
//     cv::Mat lowerThresholdedImage, higherThresholdedImage;
//     double lowerThreshold = threshold / 2.0;
//     double higherThreshold = threshold;
//     cv::threshold(diffImage, lowerThresholdedImage, lowerThreshold, 255,
//                   cv::THRESH_BINARY);
//     cv::threshold(diffImage, higherThresholdedImage, higherThreshold, 255,
//                   cv::THRESH_BINARY);

//     // Dilation to determine connectivity
//     cv::Mat dilatedHigherThresholdedImage;
//     cv::dilate(higherThresholdedImage, dilatedHigherThresholdedImage, cv::Mat());
//     cv::bitwise_and(dilatedHigherThresholdedImage, lowerThresholdedImage, output);

//     // Shadow detection
//     cv::Mat gainImage = grayFrame / background;
//     cv::Mat shadowMask = (gainImage < 1.0);

//     cv::Mat shadowMask8U;
//     shadowMask.convertTo(shadowMask8U, CV_8U);

//     cv::Mat regionLabels;
//     int numRegions = connectedComponents(shadowMask8U, regionLabels, 8, CV_32S);

//     // Array to hold the average gain in each region
//     std::vector<double> regionGains(numRegions, 0.0);

//     // Iterate over the labels to compute the average gain in each region
//     for (int i = 0; i < regionLabels.rows; i++)
//     {
//         for (int j = 0; j < regionLabels.cols; j++)
//         {
//             regionGains[regionLabels.at<int>(i, j)] += gainImage.at<double>(i, j);
//         }
//     }

//     // Override the theshold image with the shadow detection result
//     for (int i = 0; i < regionLabels.rows; i++)
//     {
//         for (int j = 0; j < regionLabels.cols; j++)
//         {
//             if (regionGains[regionLabels.at<int>(i, j)] /
//                     component_sizes[regionLabels.at<int>(i, j)] <
//                 0.5)
//             {
//                 output.at<uchar>(i, j) = 0;
//             }
//             else
//             {
//                 output.at<uchar>(i, j) = 255;
//             }
//         }
//     }

//     // Convert output to a binary image 8UC1
//     output.convertTo(output, CV_8UC1);
//     output.convertTo(output, CV_8UC3);

//     // Update the final mask
//     this->shadowMask = output;
}

void AGMM::objectExtraction()
{
    // Create a 4x4 pixel structuring element
    cv::Mat structuringElement =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));

    // Apply morphological opening to remove small isolated regions
    cv::morphologyEx(this->finalMask, this->finalMask, cv::MORPH_OPEN,
                     structuringElement);

    // Perform connected components analysis
    cv::Mat labels;
    connectedComponents(this->finalMask, labels, 8, CV_32S);

    // Create output image
    cv::Mat output = cv::Mat::zeros(this->finalMask.size(), CV_8UC1);

    // Assign 255 intensity to components (objects) on the mask
    for (int i = 0; i < labels.rows; i++)
    {
        for (int j = 0; j < labels.cols; j++)
        {
            if (labels.at<int>(i, j) > 0)
            {
                output.at<uchar>(i, j) = 255;
            }
            else
            {
                output.at<uchar>(i, j) = 0;
            }
        }
    }

    // Update the final mask
    this->finalMask = output;
}

void AGMM::objectTypeClassification()
{
    cv::Mat workingFrame;
    cvtColor(this->frame, workingFrame, cv::COLOR_BGR2GRAY);
    GaussianBlur(workingFrame, workingFrame, cv::Size(BlurSize, BlurSize), 0);

    for (unsigned int i = 0; i < this->rows; i++)
    {
        for (unsigned int j = 0; j < this->cols; j++)
        {
            double pixel = static_cast<double>(workingFrame.at<unsigned char>(i, j));
            // if pixel is not white in objectMask
            if (this->objectMask.at<unsigned char>(i, j) == 0)
                this->mixtures[i * this->cols + j].updateEta(0, pixel, this->debug);
            else if (this->objectMask.at<unsigned char>(i, j) == 255 &&
                     this->shadowMask.at<unsigned char>(i, j) == 255)
                this->mixtures[i * this->cols + j].updateEta(1, pixel, this->debug);
            else
                (this->mixtures[i * this->cols + j].updateEta(3, pixel, this->debug));
        }
    }
}