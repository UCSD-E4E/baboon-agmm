#ifndef Gaussian_H
#define Gaussian_H

#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Gaussian
{
private:
    double mean;
    double variance;
    double weight;
public:
    Gaussian(double mean, double weight);

    ~Gaussian();

    double getMean();

    double getVariance();

    double getWeight();

    void setMean(double mean);

    void setVariance(double variance);

    void setWeight(double weight);
};

#endif