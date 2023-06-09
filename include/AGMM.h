#ifndef AGMM_H
#define AGMM_H

#include "Mixture.h"
#include <opencv2/opencv.hpp>

/**
 * Class to implement the adaptive Gaussian mixture model (AGMM) algorithm.
 * The algorithm is described in the paper:
 * "Regularized Background Adaptation: A Novel Learning Rate Control Scheme for
 * Gaussian Mixture Modeling" by Horng-Horn Lin, Jen-Hui Chuang, and Tyng-Luh
 * Liu.
 */
class AGMM
{
private:
    bool debug = false;
    unsigned int rows;
    unsigned int cols;
    unsigned int fps;

    // Background maintenance parameters
    double BM_numberOfGaussians = 100;
    double BM_alpha = 0.025;
    double BM_beta_b = 0.01;
    double BM_beta_d = 1.0 / 100.0;
    double BM_beta_s = 1.0 / 900.0;
    double BM_beta_m = 1.0 / 6000.0;

    // Shadow detection parameters
    double SD_hueThreshold = 62;
    double SD_saturationThreshold = 93;
    double SD_valueUpperbound = 1;
    double SD_valueLowerbound = 0.6;

    cv::VideoCapture cap;
    cv::Mat frame;
    cv::Mat objectMask;
    cv::Mat shadowMask;
    cv::Mat finalMask;
    cv::Mat result;

    std::vector<Mixture> mixtures;

    void backgroundModelMaintenance();
    void foregroundPixelIdentification();
    void shadowDetection();
    void objectExtraction();
    void objectTypeClassification();

public:
    /**
     * Initialize the AGMM algorithm.
     * @param videoPath The path to the video file.
     */
    explicit AGMM(std::string videoPath);

    explicit AGMM(std::string videoPath, bool debug);

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
    std::vector<cv::Mat> processNextFrame();

    std::vector<double> getPixelEtas(int row, int col) const
    {
        return mixtures[row * cols + col].getEtas();
    };

    std::vector<Gaussian> getPixelGaussians(int row, int col) const
    {
        return mixtures[row * cols + col].getGaussians();
    };

    unsigned int getRows() const { return rows; };

    unsigned int getCols() const { return cols; };

    unsigned int getFPS() const { return fps; };
};

#endif