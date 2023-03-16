/**
 * @file AGMM.cpp
 * @brief Class to implement the adaptive Gaussian mixture model (AGMM) algorithm.
 * The algorithm is described in the paper:
 * "Regularized Background Adaptation: A Novel Learning Rate Control Scheme for Gaussian Mixture Modeling"
 * by Horng-Horn Lin, Jen-Hui Chuang, and Tyng-Luh Liu.
 * @author Tyler Flar
 * @version 1.0
 * @date 2023-03-14
*/
#include "AGMM.hpp"
#include <tuple>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/**
 * Initialize the model.
 * @param frame The first frame of the video.
*/
AGMM::AGMM(Mat &frame) {
    // initialize the model
    this->rows = frame.size().height;
    this->cols = frame.size().width;
    this->points = this->rows * this->cols;

    // initialize the model parameters
    this->sigmas = vector<vector<float>>(points, vector<float>(gaussians, 0));
    this->pi = vector<vector<vector<float>>>(points, vector<vector<float>>(gaussians, vector<float>(3, 0)));
    this->mu = vector<vector<vector<float>>>(points, vector<vector<float>>(gaussians, vector<float>(3, 0)));

    // initialize the initial model parameters
    this->r0 = vector<vector<vector<float>>>(points, vector<vector<float>>(gaussians, vector<float>(3, 0)));
    this->mu0 = vector<vector<float>>{vector<float>{50, 50, 50}, vector<float>{130, 130, 130}, vector<float>{210, 210, 210}};
    this->pi0 = vector<vector<float>>(this->gaussians, vector<float>{0, 0, 0});
    this->sigma0 = vector<float>(this->gaussians, 0);

    // initialize the model
    this->initializeModel(frame);

    // initialize the initial model parameters
    for (int c = 0; c < 3; c++) {
        for (int g = 0; g < this->gaussians; g++) { 
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    this->mu[j + this->cols * i][g][c] = this->mu0[g][c];
                    this->sigmas[j + this->cols * i][g] = this->sigma0[g];
                    this->pi[j + this->cols * i][g][c] = (1 / this->gaussians) * (1 - this->alpha) + this->alpha * this->r0[j + this->cols * i][g][c];
                }
            }
        }
    }
}

/**
 * Destructor.
*/
AGMM::~AGMM() {
    // nothing to do here
}

/**
 * Initialize the model.
 * @param frame The current frame of the video.
*/
void AGMM::initializeModel(Mat &frame) {
    vector<vector<float>> beta0 = this->mu0;

    // compute till convergence
    while (true) {
        // clustering each pixel of the frame to the nearest mean
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    //get pixel value for the current channel c
                    float pixelSingleChannel = frame.at<Vec3b>(i, j)[c];

                    // get this->mu0 value for the current channel c 
                    float mu0SingleChannel = this->mu0[0][c];

                    // compute distance
                    float minDistance = (pixelSingleChannel - mu0SingleChannel) * (pixelSingleChannel - mu0SingleChannel);

                    // get beta0 value for the current channel c
                    this->r0[j + this->cols * i][0][c] = 0;

                    // get the nearest mean
                    int id = 0;
                    for (int g = 1; g < this->gaussians; g++) {
                        float mu0ChangeSingleChannel = this->mu0[g][c];

                        // compute distance
                        float current = (pixelSingleChannel - mu0ChangeSingleChannel) * (pixelSingleChannel - mu0ChangeSingleChannel);

                        // get the nearest mean and distance value
                        if (current < minDistance) {
                            minDistance = current;
                            id = g;
                        }
                        
                        // assign the pixel to the nearest mean 
                        this->r0[j + this->cols * i][g][c] = 0;
                    }

                    // assign the pixel to the nearest mean
                    this->r0[j + this->cols * i][id][c] = 1;
                }
            }
        }

        // calculate the mean for the new clusters
        for (int c = 0; c < 3; c++) {
            for (int g = 0; g < this->gaussians; g++) {
                this->pi0[g][c] = 1;

                // calculate the mean for the new clusters for the current channel c and gaussian g 
                for (int i = 0; i < this->rows; i++) {
                    for (int j = 0; j < this->cols; j++) {
                        this->mu0[g][c] = this->mu0[g][c] + frame.at<Vec3b>(i, j)[c] * this->r0[j + this->cols * i][g][c];
                        this->pi0[g][c] = this->pi0[g][c] + this->r0[j + this->cols * i][g][c];
                    }
                }

                // calculate the mean for the new clusters for the current channel c and gaussian g 
                this->mu0[g][c] = this->mu0[g][c] / this->pi0[g][c];
            }
        }

        // check if the new cluster mean is converged below the threshold
        float sum = 0;
        for (int c = 0; c < 3; c++) {
            for (int g = 0; g < this->gaussians; g++) {
                sum = sum + beta0[g][c] * beta0[g][c];
                beta0[g][c] = this->mu0[g][c];
            }
        }

        // check if the new cluster mean is converged below the threshold 
        if (sum < 100) break;

        // check if the maximum number of iterations is reached
    }

    // calculate the final segmentation
    for (int g = 0; g < this->gaussians; g++) {
        this->pi0[g][0] = 0;
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                this->sigma0[g] = this->sigma0[g] + ((frame.at<Vec3b>(i, j)[0] - this->mu0[g][0]) * (frame.at<Vec3b>(i, j)[0] - this->mu0[g][0])) * this->r0[j + this->cols * i][g][0];
            }
        }

        // calculate the final segmentation for the current gaussian g 
        this->sigma0[g] = this->sigma0[g] / this->pi0[g][0];
    }
}

/**
 * Update the background model.
 * @param frame The current frame.
*/
pair<Mat, Mat> AGMM::updateBackground(Mat &frame) {
    int ratio[3] = {0};
    Mat background(this->rows, this->cols, CV_8UC3, Scalar(0, 0, 0));
    Mat foreground(this->rows, this->cols, CV_8UC3, Scalar(0, 0, 0));
 
    // update the background model for each pixel of the frame 
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->rows; j++) {
                bool belongsToGaussian = false;
                float temp = 0;
                for (int g = 0; g < this->gaussians; g++) {
                    if (abs(frame.at<Vec3d>(i, j)[c] - this->mu[j + this->cols * i][g][c]) < (2.5 * sqrt(this->sigmas[j + this->cols * i][g]))) {
                        this->mu[j + this->cols * i][g][c] = (1 - this->ro) * this->mu[j + this->cols * i][g][c] + this->ro * frame.at<Vec3d>(i, j)[c];
                        this->sigmas[j + this->cols * i][g] = (1 - ro) * this->sigmas[j + this->cols * i][g] + ro * ((frame.at<Vec3d>(i, j)[c] - this->mu[j + this->cols * i][g][c]) * (frame.at<Vec3d>(i, j)[c] - this->mu[j + this->cols * i][g][c]));
                        this->pi[j + this->cols * i][g][c] = (1 - this->alpha) * this->pi[j + this->cols * i][g][c] + this->alpha;
                        belongsToGaussian = true;
                    } else {
                        this->pi[j + this->cols * i][g][c] = (1 - this->alpha) * this->pi[j + this->cols * i][g][c];
                    }

                    // calculate the probability of the current pixel to belong to the current gaussian
                    temp = temp + this->pi[j + this->cols * i][g][c];
                }

                // calculate the probability of the current pixel to belong to the current gaussian 
                for (int g = 0; g < this->gaussians; g++) {
                    this->pi[j + this->cols * i][g][c] = this->pi[j + this->cols * i][g][c] / temp;
                    ratio[g] = this->pi[j + this->cols * i][g][c] / this->sigmas[j + this->cols * i][g];
                }

                // sort the gaussians based on the probability of the current pixel to belong to the current gaussian 
                for (int g = 0; g < this->gaussians; g++) {
                    bool swapped = false;
                    for (int h = 0; h < this->gaussians - g - 1; h++) {
                        if (ratio[h] < ratio[h + 1]) {
                            float temp = ratio[h];
                            ratio[h] = ratio[h + 1];
                            ratio[h + 1] = temp;

                            // sort the mean of the gaussians 
                            vector<float> temp2 = this->mu[j + this->cols * i][h];
                            this->mu[j + this->cols * i][h] = this->mu[j + this->cols * i][h + 1];
                            this->mu[j + this->cols * i][h + 1] = temp2;

                            // sort the variance of the gaussians
                            float temp3 = this->sigmas[j + this->cols * i][h];
                            this->sigmas[j + this->cols * i][h] = this->sigmas[j + this->cols * i][h + 1];
                            this->sigmas[j + this->cols * i][h + 1] = temp3;

                            // sort the probability of the gaussians
                            vector<float> temp4 = this->pi[j + this->cols * i][h];
                            this->pi[j + this->cols * i][h] = this->pi[j + this->cols * i][h + 1];
                            this->pi[j + this->cols * i][h + 1] = temp4;
                            
                            // sort the r of the gaussians
                            swapped = true;
                        }
                    }

                    // if no two elements were swapped by inner loop, then break 
                    if (!swapped) {
                        break;
                    }
                }

                // if the current pixel does not belong to any gaussian, add a new gaussian to the background model 
                if (!belongsToGaussian) {
                    this->mu[j + this->cols * i][this->gaussians - 1][c] = frame.at<Vec3d>(i, j)[c];
                    this->sigmas[j + this->cols * i][this->gaussians - 1] = 10000;
                }

                // Check if the current pixel is a background pixel
                float backgroundPixel = 0;
                float baseBackgroundPixel = 0;
                for (int g = 0; g < this->gaussians; g++) {
                    backgroundPixel += this->pi[j + this->cols * i][g][c];

                    // if the current pixel is a background pixel, set the background pixel to the mean of the gaussian with the highest probability
                    if (backgroundPixel > 0.9) {
                        baseBackgroundPixel = g;
                        break;
                    }
                }

                // if the current pixel is a background pixel, set the background pixel to the mean of the gaussian with the highest probability 
                for (int h = 0; h < baseBackgroundPixel + 1; h++) {
                    if (!belongsToGaussian || abs(frame.at<Vec3d>(i, j)[c] - this->mu[j + this->cols * i][h][c]) < this->sigmas[j + this->cols * i][h]) {
                        background.at<Vec3d>(i, j)[c] = this->mu[j + this->cols * i][h][c];
                        foreground.at<Vec3d>(i, j)[c] = frame.at<Vec3d>(i, j)[c];
                    } else {
                        background.at<Vec3d>(i, j)[c] = frame.at<Vec3d>(i, j)[c];
                        foreground.at<Vec3d>(i, j)[c] = 255;
                    }
                }
            }
        }
    }
    
    // set the foreground to white 
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                if (foreground.at<Vec3d>(i, j)[c] == 255) {
                    foreground.at<Vec3d>(i, j)[0] = 255;
                    foreground.at<Vec3d>(i, j)[1] = 255;
                    foreground.at<Vec3d>(i, j)[2] = 255;
                }
            }
        }
    }

    // return the foreground and background
    return make_pair(foreground, background);
}
