#ifndef Gaussian_H
#define Gaussian_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class Gaussian
{
private:
    double mean;
    double variance;
    double weight;

public:
    Gaussian(double mean, double varience, double weight);

    ~Gaussian();

    double getMean() const;

    double getVariance() const;

    double getWeight() const;

    void setMean(double mean);

    void setVariance(double variance);

    void setWeight(double weight);
};

#endif