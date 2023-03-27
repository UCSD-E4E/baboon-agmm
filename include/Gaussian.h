#ifndef Gaussian_H
#define Gaussian_H

#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * Class to represent a Gaussian component.
*/
class Gaussian {
    private:
        // mean values
        double meanB;
        double meanG;
        double meanR;

        // standard deviations
        double variance;
        double weight;
        double weightDistrRatio;

        double lowerboundVariance;
        double upperboundVariance;

        double calculateVariance(vector<Vec3b> pixels, double mean);

    public:
        /**
         * Initialize the Gaussian component.
        */
        Gaussian(vector<Vec3b> samples, double meanB, double meanG, double meanR, double lowerboundVariance, double upperboundVariance, int numberOfGaussians);

        Gaussian(double meanB, double meanG, double meanR, double lowerboundVariance, double upperboundVariance, double weight);

        ~Gaussian();

        double getProbablity(double distance);

        double getMeanB();

        double getMeanG();

        double getMeanR();

        double getVariance();

        double getWeight();

        double getWeightDistrRatio();

        void setMeanB(double meanB);

        void setMeanG(double meanG);

        void setMeanR(double meanR);

        void setVariance(double variance);

        void setWeight(double weight);

        void setWeightDistrRatio(double weightDistrRatio);
};

#endif