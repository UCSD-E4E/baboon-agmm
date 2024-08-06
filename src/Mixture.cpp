#include "../include/Mixture.h"
#include "../include/Gaussian.h"
#include <limits>
#include <opencv2/opencv.hpp>
#include <random>

const double weightThreshold = .24;

Mixture::Mixture(int numberOfGaussians, double alpha, double beta_b,
                 double beta_d, double beta_s, double beta_m)
{
    this->numberOfGaussians = numberOfGaussians;
    this->alpha = alpha;
    this->beta_b = beta_b;
    this->beta_d = beta_d;
    this->beta_s = beta_s;
    this->beta_m = beta_m;
}

Mixture::~Mixture()
{
    // Nothing to do here
}

double calculateProbability(double intensity, double mean, double variance)
{
    double probability = (1.0 / sqrt(2.0 * M_PI * variance)) *
                         (exp(-pow(intensity - mean, 2.0) / (2.0 * variance)));
    return probability;
}

void Mixture::initializeMixture(double intensity)
{
    // Initialize this->gaussians with this->numberOfGaussians Gaussian
    // distributions. Mean is intensity, variance is 100, and weight is 1 /
    // this->numberOfGaussians
    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        Gaussian gaussian(intensity, 100.0,
                          1.0 / static_cast<double>(this->numberOfGaussians));
        this->gaussians.push_back(gaussian);
    }

    this->etas.push_back(this->eta);
}

void Mixture::updateMixture(double intensity)
{
    // Model matching
    // M_{t,x,n} = 0, for some n = 1, ..., N
    std::vector<int> modelMatching(this->numberOfGaussians, 0);
    // d_{t,x,n} = infinity, for some n = 1, ..., N
    std::vector<double> distances(this->numberOfGaussians,
                                  std::numeric_limits<double>::infinity());

    // for n = 1,...,N do
    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        // if |I_{t,x} - mu_{t-1,x,n}| <= 2.5 * sigma_{t-1,x,n} then d_{t,x,n} =
        // -w_{t-1,x,n}
        double standardDeviation = sqrt(this->gaussians[n].getVariance());
        if (abs(intensity - this->gaussians[n].getMean()) <=
            2.5 * standardDeviation)
        {
            distances[n] = -this->gaussians[n].getWeight();
        }
    }

    // l(t,x) = argmin_{n=1,...,N} d_{t,x,n}
    int currentGaussianIndex = distance(
        distances.begin(), min_element(distances.begin(), distances.end()));

    // if d_{t,x,l(t,x)} != inf then M_{t,x,l(t,x)} = 1 else l(t,x) = 0
    if (distances[currentGaussianIndex] !=
        std::numeric_limits<double>::infinity())
    {
        modelMatching[currentGaussianIndex] = 1;
    }
    else
    {
        currentGaussianIndex = 0;
    }

    // Model renewing
    // w_{t,x,n} = (1 - eta_{t,x}(Beta)) * w_{t-1,x,n} + eta_{t,x}(Beta) *
    // M_{t,x,n}, for some n
    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        this->gaussians[n].setWeight(
            (1.0 - this->eta) * this->gaussians[n].getWeight() +
            this->eta * static_cast<double>(modelMatching[n]));
    }

    // if M_{t,x,l(t,x)} = 1
    if (modelMatching[currentGaussianIndex] == 1)
    {
        // Update phase
        // rho_{t,x,l(t,x)}(alpha) = alpha * GF(I_{t,x}, mu_{t-1,x,l(t,x)},
        // sigma^2_{t-1,x,l(t,x)})
        double palpha =
            this->alpha * calculateProbability(
                              intensity,
                              this->gaussians[currentGaussianIndex].getMean(),
                              this->gaussians[currentGaussianIndex].getVariance());
        // mu_{t,x,l(t,x)} = (1 - rho_{t,x,l(t,x)}(alpha)) * mu_{t-1,x,l(t,x)} +
        // rho_{t,x,l(t,x)}(alpha) * I_{t,x}
        this->gaussians[currentGaussianIndex].setMean(
            (1.0 - palpha) * this->gaussians[currentGaussianIndex].getMean() +
            palpha * intensity);
        // sigma^2_{t,x,l(t,x)} = (1 - rho_{t,x,l(t,x)}(alpha)) *
        // sigma^2_{t-1,x,l(t,x)} + rho_{t,x,l(t,x)}(alpha) * (I_{t,x} -
        // mu_{t,x,l(t,x)})^2
        this->gaussians[currentGaussianIndex].setVariance(
            (1.0 - palpha) * this->gaussians[currentGaussianIndex].getVariance() +
            palpha *
                pow(intensity - this->gaussians[currentGaussianIndex].getMean(),
                    2.0));
    }
    else
    {
        // Replacement phase
        // k = argmin_{n=1,...,N} w_{t-1,x,n}
        int gaussianToReplaceIndex =
            distance(this->gaussians.begin(),
                     min_element(this->gaussians.begin(), this->gaussians.end(),
                                 [](const Gaussian &g1, const Gaussian &g2)
                                 {
                                     return g1.getWeight() < g2.getWeight();
                                 }));
        // mu_{t,x,k} = I_{t,x}
        this->gaussians[gaussianToReplaceIndex].setMean(intensity);
        // sigma^2_{t,x,k} = = sigma^2_0
        this->gaussians[gaussianToReplaceIndex].setVariance(100);
        // w_{t,x,k} = w_0
        this->gaussians[gaussianToReplaceIndex].setWeight(
            1.0 / (double)this->numberOfGaussians);
    }

    // w_{t,x,n} = w_{t,x,n} / sum_{n=1}^N w_{t,x,n}, for some n
    double sumOfWeights = 0;
    for (int n = 0; n < this->numberOfGaussians; n++)
    {
        sumOfWeights += this->gaussians[n].getWeight();
    }

    if (sumOfWeights == 0)
    {
        throw std::runtime_error("sumOfWeights is zero, cannot divide by zero");
    }
    else
    {
        for (int n = 0; n < this->numberOfGaussians; n++)
        {
            this->gaussians[n].setWeight(this->gaussians[n].getWeight() /
                                         sumOfWeights);
        }
    }
}

bool Mixture::isForegroundPixel() const
{
    // F_{t,x,n} = {0, if w_{t,x,n} >= T_w, 1, otherwise}, where n is the heighest
    // weighted Gaussian
    int currentGaussianIndex =
        distance(this->gaussians.begin(),
                 max_element(this->gaussians.begin(), this->gaussians.end(),
                             [](const Gaussian &g1, const Gaussian &g2)
                             {
                                 return g1.getWeight() < g2.getWeight();
                             }));
    return this->gaussians[currentGaussianIndex].getWeight() < weightThreshold;
}

void Mixture::updateEta(int O, double intensity, bool debug)
{
    // eta_{t,x}(Beta) =
    switch (O)
    {
    case 0:
    {
        // (1 - Beta_b) * eta_{t-1,x} + eta_{b}*Beta_b
        this->eta = (1.0 - this->beta_b) * this->eta + 0.025 * this->beta_b;
        break;
    }
    case 1:
    {
        // b(t,x) = argmax_{n=1,...,N} w_{t,x,n}
        int currentGaussianIndex =
            distance(this->gaussians.begin(),
                     max_element(this->gaussians.begin(), this->gaussians.end(),
                                 [](const Gaussian &g1, const Gaussian &g2)
                                 {
                                     return g1.getWeight() < g2.getWeight();
                                 }));
        // Beta_d * GF(I_{t,x}, mu_{t,x,b(t,x)}, sigma^2_{t,x,b(t,x)})
        this->eta = this->beta_d *
                    calculateProbability(
                        intensity, this->gaussians[currentGaussianIndex].getMean(),
                        this->gaussians[currentGaussianIndex].getVariance());
        // Keep below Beta_b
        if (this->eta > this->beta_b)
        {
            this->eta = this->beta_b;
        }

        // If the product is less than Beta_s, reset eta to Beta_s
        if (this->eta < this->beta_s)
        {
            this->eta = this->beta_s;
        }
        break;
    }
    case 2:
    {
        // B_s
        this->eta = this->beta_s;
        break;
    }
    case 3:
    {
        // B_m
        this->eta = this->beta_m;
        break;
    }
    }

    if (debug)
        this->etas.push_back(this->eta);
}
