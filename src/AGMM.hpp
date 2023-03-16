#ifndef AGMM_H
#define AGMM_H

#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;   

/**
 * Class to implement the adaptive Gaussian mixture model (AGMM) algorithm.
 * The algorithm is described in the paper:
 * "Regularized Background Adaptation: A Novel Learning Rate Control Scheme for Gaussian Mixture Modeling"
 * by Horng-Horn Lin, Jen-Hui Chuang, and Tyng-Luh Liu.
*/
class AGMM {
    private: 
        int rows; // number of rows in the frame
        int cols; // number of columns in the frame
        int points; // number of points in the frame
        int gaussians = 3; // number of Gaussian components
        float alpha = 0.025; // learning rate
        float ro = 0.11; // regularization parameter
        
        vector<vector<float>> sigmas; // standard deviations
        vector<vector<vector<float>>> pi; // weights
        vector<vector<vector<float>>> mu; // means

        vector<vector<float>> pi0; // initial weights 
        vector<vector<vector<float>>> r0; // initial responsibilities
        vector<vector<float>> mu0; // initial means
        vector<float> sigma0; // initial standard deviations

    public:
        /**
         * Initialize the model.
         * @param frame The first frame of the video.
        */
        explicit AGMM(Mat &frame);

        /**
         * Helper to initialize the model.
        */
        void initializeModel(Mat &frame);

        /**
         * Update the model.
         * @param frame The current frame of the video.
         * @return A pair of vectors of vectors of tuples. The first vector of vectors of tuples contains the foreground pixels.
        */
        pair<Mat,Mat> updateBackground(Mat &frame);
};

#endif