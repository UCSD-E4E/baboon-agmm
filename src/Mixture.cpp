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

void Mixture::initializeMixture(vector<Vec3b> pixels)
{
    // Initialize the Gaussian components
    for (int i = 0; i < this->numberOfGaussians; i++)
    {
        Gaussian gaussian = Gaussian(0, 0, 0);
        this->gaussians.push_back(gaussian);
    }

    // Randomly select pixels
    random_shuffle(pixels.begin(), pixels.end());
    pixels.resize(this->numberOfGaussians);

    // Initiialize the mean and varience of the Gaussian components using the pixel intensities
    for (int k = 0; k < this->numberOfGaussians; k++)
    {
        double mean = 0;
        double variance = 0;

        mean = (pixels[k][0] + pixels[k][1] + pixels[k][2]) / 3;

        for (long unsigned int i = 0; i < pixels.size(); i++)
        {
            double diff = (pixels[i][0] + pixels[i][1] + pixels[i][2]) / 3 - mean;
            variance += diff * diff;
        }

        variance /= pixels.size();

        // Set the mean and variance of the Gaussian component
        this->gaussians[k].setMean(mean);
        this->gaussians[k].setVariance(variance);
    }

    // Set the weights of the Gaussian compenents
    for (int k = 0; k < this->numberOfGaussians; k++)
    {
        this->gaussians[k].setWeight(1.0 / this->numberOfGaussians);
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

bool Mixture::isForegroundPixel(Vec3b pixel)
{
    // Calculate the intensity of the pixel
    double intensity = (pixel[0] + pixel[1] + pixel[2]) / 3;

    // Calculate the mixture probabilities for each Gaussian
    double mixtureProbabilities[this->numberOfGaussians];
    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        mixtureProbabilities[n] = this->gaussians[n].getWeight() * (1 / sqrt(2 * M_PI * this->gaussians[n].getVariance())) * exp(-pow(intensity - this->gaussians[n].getMean(), 2) / (2 * this->gaussians[n].getVariance()));
    }

    // Calculate the foreground probability using the two Gaussians with the highest weights
    double maxWeight1 = 0;
    double maxWeight2 = 0;
    int maxIndex1 = 0;
    int maxIndex2 = 0;
    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        if (this->gaussians[n].getWeight() > maxWeight1)
        {
            maxWeight2 = maxWeight1;
            maxIndex2 = maxIndex1;
            maxWeight1 = this->gaussians[n].getWeight();
            maxIndex1 = n;
        }
        else if (this->gaussians[n].getWeight() > maxWeight2)
        {
            maxWeight2 = this->gaussians[n].getWeight();
            maxIndex2 = n;
        }
    }

    double foregroundProbability = mixtureProbabilities[maxIndex1] / (mixtureProbabilities[maxIndex1] + mixtureProbabilities[maxIndex2]);

    // Return true if the pixel is classified as foreground, false otherwise
    return (foregroundProbability > .24);
}
