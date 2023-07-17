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
    bool disableShadow = false;
    unsigned int rows;
    unsigned int cols;
    unsigned int fps;

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

    explicit AGMM(std::string videoPath, bool debug, bool disableShadow);

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