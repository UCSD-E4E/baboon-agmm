#ifndef Mixture_H
#define Mixture_H

#include "Gaussian.h"
#include <opencv2/opencv.hpp>

class Mixture
{
private:
    int numberOfGaussians;

    int frameNumber = 0;
    double eta = .025;
    double alpha;
    double beta_b;
    double beta_d;
    double beta_s;
    double beta_m;

    double calculateProbablility(double intensity, double mean, double variance);

    std::vector<double> etas;
    std::vector<Gaussian> gaussians;

public:
    Mixture(int numberOfGaussians, double alpha, double beta_b, double beta_s,
            double beta_sf, double beta_mf);

    ~Mixture();

    void initializeMixture(double intensity);

    void updateMixture(double intensity);

    bool isForegroundPixel() const;

    void updateEta(int O, double intensity, bool debug);

    std::vector<double> getEtas() const { return this->etas; };

    std::vector<Gaussian> getGaussians() const { return this->gaussians; };
};

#endif