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
class AGMM
{
private:
    // Background maintenance parameters
    double BM_numberOfGaussians = 100;
    double BM_alpha = 0.025;
    double BM_beta_b = 0.01;
    double BM_beta_d = 1.0/100.0;
    double BM_beta_s = 1.0/900.0;
    double BM_beta_m = 1.0/6000.0;

    // Shadow detection parameters
    double SD_hueThreshold = 62;
    double SD_saturationThreshold = 93;
    double SD_valueUpperbound = 1;
    double SD_valueLowerbound = 0.6;

    VideoCapture cap;
    Mat frame;
    Mat background;
    Mat objectMask;
    Mat shadowMask;
    Mat finalMask;
    Mat result;



    vector<Mixture> mixtures;

    void backgroundModelMaintenance();
    void foregroundPixelIdentification();
    void shadowDetection();
    void objectExtraction();
    void objectTypeClassification();

public:
    unsigned int rows;
    unsigned int cols;

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
    void initializeModel();

    /**
     * Process the next frame and return the foreground mask.
     * @return The foreground mask.
     */
    tuple<Mat, Mat, Mat> processNextFrame();

    vector<double> getPixelEtas(int row, int col);

    vector<Gaussian> getPixelGaussians(int row, int col);
};

#endif