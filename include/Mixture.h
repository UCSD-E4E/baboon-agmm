#ifndef Mixture_H
#define Mixture_H

#include "Gaussian.h"
#include <opencv2/opencv.hpp>

using namespace cv;

class Mixture
{
private:
    int numberOfGaussians;

    int O = 0;
    double eta = 1/6000;
    double alpha;
    double beta_b;
    double beta_s;
    double beta_sf;
    double beta_mf;;

    vector<Gaussian> gaussians;

public:
    Mixture(double numberOfGaussians, double alpha, double beta_b, double beta_s, double beta_sf, double beta_mf);

    ~Mixture();

    void initializeMixture();
    
    void updateMixture(double intensity);

    bool isForegroundPixel();
};

#endif