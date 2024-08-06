#ifndef Gaussian_H
#define Gaussian_H

class Gaussian
{
private:
    double mean;
    double variance;
    double weight;

public:
    Gaussian(double mean, double variance, double weight)
        : mean(mean), variance(variance), weight(weight) {}

    ~Gaussian() = default;

    double getMean() const { return mean; }

    double getVariance() const { return variance; }

    double getWeight() const { return weight; }

    void setMean(double mean) { this->mean = mean; }

    void setVariance(double variance) { this->variance = variance; }

    void setWeight(double weight) { this->weight = weight; }
};

#endif