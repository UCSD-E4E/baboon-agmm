#include "../include/Mixture.h"
#include "../include/Gaussian.h"
#include <random>
#include <limits>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mixture::Mixture(double numberOfGaussians, double alpha, double beta_b, double beta_s, double beta_sf, double beta_mf)
{
    this->numberOfGaussians = numberOfGaussians;
    this->alpha = alpha;
    this->beta_b = beta_b;
    this->beta_s = beta_s;
    this->beta_sf = beta_sf;
    this->beta_mf = beta_mf;
}

Mixture::~Mixture()
{
    this->gaussians.clear();
}

void Mixture::initializeMixture(Vec3b pixel)
{
    // Initialize the Gaussian components
    for (int i = 0; i < this->numberOfGaussians; i++)
    {
        if (i == 0) {
            double mean = (pixel[0] + pixel[1] + pixel[2]) / 3;
            Gaussian gaussian(mean, 1, 1);
            this->gaussians.push_back(gaussian);
        }
        else 
        {
            Gaussian gaussian(0, 10*10, 0);
            this->gaussians.push_back(gaussian);
        }


    }
}

void Mixture::updateMixture(Vec3b pixel)
{
    // Calculate the intensity of the pixel
    double intensity = (pixel[0] + pixel[1] + pixel[2]) / 3;
    double modelMatching[this->numberOfGaussians]{0};
    double distances[this->numberOfGaussians];

    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        // set distances to infinity
        distances[n] = double(numeric_limits<double>::infinity());
    }

    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        if (abs(intensity - this->gaussians[n].getMean()) <= 2.5 * this->gaussians[n].getVariance())
        {
            distances[n] = -this->gaussians[n].getWeight();
        }
    }

    // Find the minimum distance
    double minDistance = numeric_limits<double>::infinity();
    int minDistanceIndex = 0;
    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        if (distances[n] < minDistance)
        {
            minDistance = distances[n];
            minDistanceIndex = n;
        }
    }

    if (distances[minDistanceIndex] != numeric_limits<double>::infinity())
    {
        modelMatching[minDistanceIndex] = 1;
    }
    else
    {
        minDistanceIndex = -1;
    }

    switch (this->O)
    {
    case 0:
    {
        this->eta = (1 - this->beta_b) * this->eta + .025 * this->beta_b;
        break;
    }
    case 1:
    {
        double maxDistanceIndex = 0;
        for (int n = 0; n < this->numberOfGaussians; n++)
        {
            if (abs(intensity - this->gaussians[n].getMean()) > abs(intensity - this->gaussians[maxDistanceIndex].getMean()))
            {
                maxDistanceIndex = n;
            }
        }

        this->eta = (1 / sqrt(2 * M_PI * this->gaussians[maxDistanceIndex].getVariance())) * exp(-(pow(intensity - gaussians[maxDistanceIndex].getMean(), 2)) / (2 * this->gaussians[maxDistanceIndex].getVariance()));
        break;
    }
    case 2:
    {
        this->eta = beta_sf;
        break;
    }
    case 3:
    {
        this->eta = beta_mf;
        break;
    }
    }

    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        this->gaussians[n].setWeight((1 - this->eta) * this->gaussians[n].getWeight() + this->eta * modelMatching[n]);
    }

    if (modelMatching[minDistanceIndex] == 1)
    {
        // Update phase
        double palpha = this->alpha * (1 / sqrt(2 * M_PI * this->gaussians[minDistanceIndex].getVariance())) * exp(-pow(intensity - this->gaussians[minDistanceIndex].getMean(), 2) / (2 * this->gaussians[minDistanceIndex].getVariance()));

        this->gaussians[minDistanceIndex].setMean((1 - palpha) * this->gaussians[minDistanceIndex].getMean() + palpha * intensity);
        this->gaussians[minDistanceIndex].setVariance((1 - palpha) * this->gaussians[minDistanceIndex].getVariance() + palpha * pow(intensity - this->gaussians[minDistanceIndex].getMean(), 2));
    }
    else
    {
        // Replacement phase
        double minWeight = numeric_limits<double>::infinity();
        int minWeightIndex = 0;
        for (int n = 0; n < this->numberOfGaussians; n++)
        {
            if (this->gaussians[n].getWeight() < minWeight)
            {
                minWeight = this->gaussians[n].getWeight();
                minWeightIndex = n;
            }
        }

        this->gaussians[minWeightIndex].setMean(intensity);
        this->gaussians[minWeightIndex].setVariance(pow(10, 2));
        this->gaussians[minWeightIndex].setWeight(0.01);
    }

     double sum = 0;
    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        sum += this->gaussians[n].getWeight();
    }

    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        this->gaussians[n].setWeight(this->gaussians[n].getWeight() / sum);
    }
}

bool Mixture::isForegroundPixel()
{
    bool isForeground = false;
    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        if (this->gaussians[n].getWeight() >= .9)
        {
            isForeground = true;
            break;
        }
    }

    return isForeground;
}
