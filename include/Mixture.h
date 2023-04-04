#ifndef Mixture_H
#define Mixture_H

#include "Gaussian.h"
#include <opencv2/opencv.hpp>

using namespace cv;

class Mixture
{
private:
    int numberOfGaussians;

    double alpha;
    double upperboundVariance;
    double lowerboundVariance;

    vector<Gaussian> gaussians;

    vector<Vec3b> randomSamplePixel(vector<Vec3b> pixels, int index);

public:
    /**
     * Create Mixture Object.
     * @param numberOfGaussians The number of Gaussian components.
     * @param alpha The learning rate.
     * @param upperboundVariance The upper bound of the variance.
     * @param lowerboundVariance The lower bound of the variance.
     */
    Mixture(double numberOfGaussians, double alpha, double upperboundVariance, double lowerboundVariance);

    ~Mixture();

    /**
     * Initialize the mixture.
     * @param pixels The pixels to initialize the mixture with.
     */
    void initializeMixture(vector<Vec3b> pixels);

    /**
     * Update the mixture.
     * @param pixel The pixel to update the mixture with.
     * @param threshold The threshold to use for updating the mixture.
     * @return True if mixture is foreground, false if mixture is background.
     */
    bool updateMixture(Vec3b pixel, double threshold);

    int getNumberOfGaussians();

    double getAlpha();

    double getUpperboundVariance();

    double getLowerboundVariance();

    vector<Gaussian> getGaussians();

    void setNumberOfGaussians(int numberOfGaussians);

    void setAlpha(double alpha);

    void setUpperboundVariance(double upperboundVariance);

    void setLowerboundVariance(double lowerboundVariance);

    void setGaussians(vector<Gaussian> gaussians);
};

#endif