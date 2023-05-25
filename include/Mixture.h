#ifndef Mixture_H
#define Mixture_H

#include "Gaussian.h"
#include <opencv2/opencv.hpp>

using namespace cv;

class Mixture
{
private:
    int numberOfGaussians;

    double eta = .025;
    double alpha;
    double beta_b;
    double beta_d;
    double beta_s;
    double beta_m;;

    vector<Gaussian> gaussians;

    double calculateProbablility(double intensity, double mean, double variance);

public:
    Mixture(double numberOfGaussians, double alpha, double beta_b, double beta_s, double beta_sf, double beta_mf);

    ~Mixture();

    void initializeMixture(double intensity);
    
    void updateMixture(double intensity);

    bool isForegroundPixel();

    void updateEta(int O, double intensity);
};

#endif