#include "../include/Gaussian.h"

Gaussian::Gaussian(double mean, double varience, double weight)
{
    this->mean = mean;
    this->variance = varience;
    this->weight = weight;
}

Gaussian::~Gaussian()
{
}


double Gaussian::getMean()
{
    return this->mean;
}

double Gaussian::getVariance()
{
    return this->variance;
}

double Gaussian::getWeight()
{
    return this->weight;
}


void Gaussian::setMean(double mean)
{
    this->mean = mean;
}

void Gaussian::setVariance(double variance)
{
    this->variance = variance;
}

void Gaussian::setWeight(double weight)
{
    this->weight = weight;
}
