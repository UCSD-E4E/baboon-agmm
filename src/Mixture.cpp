#include "../include/Mixture.h"
#include "../include/Gaussian.h"
#include <random>
#include <limits>
#include <opencv2/opencv.hpp>
#include <omp.h>

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

void Mixture::initializeMixture()
{
    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        Gaussian gaussian = Gaussian(rand() % 255, 100, 1.0 / (double)this->numberOfGaussians);
        this->gaussians.push_back(gaussian);
    }
}

void Mixture::updateMixture(double intensity)
{
    // vector<int> modelMatching(this->numberOfGaussians, 0);
    // vector<double> distances(this->numberOfGaussians, numeric_limits<double>::infinity());

    // for (int n = 0; n < this->numberOfGaussians; n++)
    // {
    //     double standardDeviation = sqrt(this->gaussians[n].getVariance());
    //     if (abs(intensity - this->gaussians[n].getMean()) <= 2.5 * standardDeviation)
    //     {
    //         distances[n] = -this->gaussians[n].getWeight();
    //     }
    // }

    // int minDistanceIndex = distance(distances.begin(), min_element(distances.begin(), distances.end()));

    // if (distances[minDistanceIndex] != numeric_limits<double>::infinity())
    // {
    //     modelMatching[minDistanceIndex] = 1;
    // }

    bool modelMatch = (abs(intensity - this->gaussians[this->numberOfGaussians - 1].getMean()) <= 2.5 * sqrt(this->gaussians[this->numberOfGaussians - 1].getVariance()));

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
    case 4:
    {
        this->eta = .01;
        break;
    }
    }

    #pragma omp parallel for num_threads(8)
    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        double newWeight = (1 - this->eta) * this->gaussians[n].getWeight();

        if (n == this->numberOfGaussians - 1 && modelMatch)
        {
            newWeight += this->eta;
        }

        this->gaussians[n].setWeight(newWeight);
    }

    if (modelMatch)
    {
        // Update phase
        double min_variance = this->gaussians[this->numberOfGaussians - 1].getVariance();
        double min_mean = this->gaussians[this->numberOfGaussians - 1].getMean();
        double palpha = this->alpha * (1 / sqrt(2 * M_PI * min_variance)) * exp(-pow(intensity - min_mean, 2) / (2 * min_variance));

        this->gaussians[this->numberOfGaussians - 1].setMean((1 - palpha) * this->gaussians[this->numberOfGaussians - 1].getMean() + palpha * intensity);
        this->gaussians[this->numberOfGaussians - 1].setVariance((1 - palpha) * this->gaussians[this->numberOfGaussians - 1].getVariance() + palpha * pow(intensity - this->gaussians[this->numberOfGaussians - 1].getMean(), 2));
    }
    else
    {

        this->gaussians[0].setMean(intensity);
        this->gaussians[0].setVariance(100);
        this->gaussians[0].setWeight(1.0 / this->numberOfGaussians);
    }

    double sum_weights = accumulate(this->gaussians.begin(), this->gaussians.end(), 0.0,
                                    [](double sum, const Gaussian &g)
                                    {
                                        return sum + g.getWeight();
                                    });

    for (auto &gaussian : this->gaussians)
    {
        gaussian.setWeight(gaussian.getWeight() / sum_weights);
    }

    // sort gaussians from smallest weight to largest weight
    sort(this->gaussians.begin(), this->gaussians.end(), [](const Gaussian &g1, const Gaussian &g2)
         { return g1.getWeight() < g2.getWeight(); });
}

bool Mixture::isForegroundPixel()
{
    // double maxWeight = 0;
    // for (int n = 0; n < this->numberOfGaussians; n++)
    // {
    //     if (this->gaussians[n].getWeight() > maxWeight)
    //         maxWeight = this->gaussians[n].getWeight();
    // }

    if (this->gaussians[this->numberOfGaussians - 1].getWeight() >= .24)
    {
        return false;
    }
    else
    {
        return true;
    }
}
