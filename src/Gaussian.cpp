#include "../include/Gaussian.h"

Gaussian::Gaussian(vector<Vec3b> samples, double meanB, double meanG, double meanR, double lowerboundVariance, double upperboundVariance, int numberOfGaussians) {
    this->meanB = meanB;
    this->meanG = meanG;
    this->meanR = meanR;
    this->lowerboundVariance = lowerboundVariance;
    this->upperboundVariance = upperboundVariance;
    
    this->variance = calculateVariance(samples, (meanB + meanG + meanR) / 3);
    

    if (this->variance < lowerboundVariance) {
        this->variance = lowerboundVariance;
    }

    this->weight = 1.0 / numberOfGaussians;
    this->weightDistrRatio = this->weight / sqrt(this->variance);
}

Gaussian::Gaussian(double meanB, double meanG, double meanR, double lowerboundVariance, double upperboundVariance, double weight) {
    this->meanB = meanB;
    this->meanG = meanG;
    this->meanR = meanR;
    this->lowerboundVariance = lowerboundVariance;
    this->upperboundVariance = upperboundVariance;
    this->variance = upperboundVariance;
    this->weight = weight;
    this->weightDistrRatio = this->weight / sqrt(this->variance);
}

Gaussian::~Gaussian() {
}

double Gaussian::calculateVariance(vector<Vec3b> pixels, double mean) {
    double variance = 0;

    for (unsigned int i = 0; i < pixels.size(); i++) {
        variance += (pixels[i][0] - mean) + (pixels[i][1] - mean) + (pixels[i][2] - mean);
    }

    return variance / pixels.size();
}

double Gaussian::getProbablity(double distance) {
    return (1 / sqrt(2 * M_PI * this->variance)) * exp(-distance / (2 * this->variance));
}

double Gaussian::getMeanB() {
    return this->meanB;
}

double Gaussian::getMeanG() {
    return this->meanG;
}

double Gaussian::getMeanR() {
    return this->meanR;
}

double Gaussian::getVariance() {
    return this->variance;
}

double Gaussian::getWeight() {
    return this->weight;
}

double Gaussian::getWeightDistrRatio() {
    return this->weightDistrRatio;
}

void Gaussian::setMeanB(double meanB) {
    this->meanB = meanB;
}

void Gaussian::setMeanG(double meanG) {
    this->meanG = meanG;
}

void Gaussian::setMeanR(double meanR) {
    this->meanR = meanR;
}

void Gaussian::setVariance(double variance) {
    this->variance = variance;
}

void Gaussian::setWeight(double weight) {
    this->weight = weight;
}

void Gaussian::setWeightDistrRatio(double weightDistrRatio) {
    this->weightDistrRatio = weightDistrRatio;
}