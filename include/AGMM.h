#ifndef AGMM_H
#define AGMM_H

#include "Mixture.h"
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
        double numberOfGaussians = 7;
        double alpha = 0.001;
        double backgroundRatio = 0.9;
        double upperboundVariance = 64;
        double lowerboundVariance = 16;
        
        VideoCapture cap;

        unsigned int rows;
        unsigned int cols;
        unsigned int numberOfPixels;

        vector<Mixture> mixtures;

    public:
        /**
         * Initialize the AGMM algorithm.
         * @param videoPath The path to the video file.
        */
        explicit AGMM(string videoPath);

        ~AGMM();

        /**
         * Initialize the model.
         * @param numberOfFrames The number of frames to use for initialization.
        */
        void initializeModel(int numberOfFrames);

        /**
         * Process the next frame and return the foreground mask. 
         * @return The foreground mask.
        */
        tuple<Mat, Mat, Mat> processNextFrame();
};


#endif