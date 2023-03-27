#include "../include/Mixture.h"
#include "../include/Gaussian.h"
#include <random>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mixture::Mixture(double numberOfGaussians, double alpha, double upperboundVariance, double lowerboundVariance) {
    this->numberOfGaussians = numberOfGaussians;
    this->alpha = alpha;
    this->upperboundVariance = upperboundVariance;
    this->lowerboundVariance = lowerboundVariance;
}

Mixture::~Mixture() {
    gaussians.clear();
}

vector<Vec3b> Mixture::randomSamplePixel(vector<Vec3b> pixels, int N) {
    random_device rd;
    mt19937 eng(rd());
    uniform_int_distribution<> generator(0, pixels.size() - 1);

    vector<Vec3b> randomPixel;
    vector<int> usedIndices;

    // Generated N random indices
    for (int i = 0; i < N; i++) {
        int index = generator(eng);

        // Check if the index is already used
        if (find(usedIndices.begin(), usedIndices.end(), index) != usedIndices.end()) {
            i--;
        } else {
            usedIndices.push_back(index);
            randomPixel.push_back(pixels[index]);
        }
    }

    return randomPixel;
}



void Mixture::initializeMixture(vector<Vec3b> pixels) {
    // Randomly sample N pixels from the image
    vector<Vec3b> randomPixels = randomSamplePixel(pixels, this->numberOfGaussians);

    // Initialize the Gaussian components
    for (int i = 0; i < this->numberOfGaussians; i++) {
        Vec3b pixel = randomPixels[i];

        double meanB = static_cast<double>(pixel[0]);
        double meanG = static_cast<double>(pixel[1]);
        double meanR = static_cast<double>(pixel[2]);

        Gaussian gaussian(randomPixels, meanB, meanG, meanR, this->lowerboundVariance, this->upperboundVariance, this->numberOfGaussians);
        this->gaussians.push_back(gaussian);
    }

    // Sort the Gaussian components by their weightRatio
    sort(this->gaussians.begin(), this->gaussians.end(), [](Gaussian a, Gaussian b) {
        return a.getWeightDistrRatio() > b.getWeightDistrRatio();
    });
}

bool Mixture::updateMixture(Vec3b pixel, double threshold) {
    bool isBackgrond = false;

    // Find index of mixture until which we consider background disttributions. Asumming the order is in decnding order
    int index = 0;
    double sum = 0;
    for (int i = 0; i < this->numberOfGaussians; i++) {
        if (sum < threshold) {
            sum += this->gaussians[i].getWeightDistrRatio();
            index++;
        } else {
            break;
        }
    }

    bool found = false;
    double weightSum = 0;
    for (int i = 0; i < this->numberOfGaussians; i++) {
        double weight = this->gaussians[i].getWeight();
        double meanB = this->gaussians[i].getMeanB();
        double meanG = this->gaussians[i].getMeanG();
        double meanR = this->gaussians[i].getMeanR();
        double variance = this->gaussians[i].getVariance();

        double distance = pow(meanB  - pixel[0], 2) + pow(meanG - pixel[1], 2) + pow(meanR - pixel[2], 2);

        if (distance < 7.5 * variance && i < index) {
            isBackgrond = true; 
        }

        if (found) {
            this->gaussians[i].setWeight(weight * (1 - this->alpha));
            if (this->gaussians[i].getWeight() < 0.0001) {
                this->gaussians[i].setWeight(0.0001);
            }
        } else if (distance < 3 * variance) {
            this->gaussians[i].setWeight((1 - this->alpha) * weight + this->alpha);
            double probability = this->gaussians[i].getProbablity(distance);

            this->gaussians[i].setMeanB((1 - this->alpha) * meanB + probability * static_cast<double>(pixel[0]));
            this->gaussians[i].setMeanG((1 - this->alpha) * meanG + probability * static_cast<double>(pixel[1]));
            this->gaussians[i].setMeanR((1 - this->alpha) * meanR + probability * static_cast<double>(pixel[2]));
            this->gaussians[i].setVariance((1 - this->alpha) * variance + probability * (distance - variance));

            if (this->gaussians[i].getVariance() < this->lowerboundVariance) {
                this->gaussians[i].setVariance(this->lowerboundVariance);
            } else if (this->gaussians[i].getVariance() > 5*this->upperboundVariance) {
                this->gaussians[i].setVariance(5*this->upperboundVariance);
            }
        }

        weightSum += this->gaussians[i].getWeight();
    }

    if(!found) {
        Gaussian gaussian(static_cast<double>(pixel[0]), static_cast<double>(pixel[1]), static_cast<double>(pixel[2]), this->lowerboundVariance, this->upperboundVariance, this->gaussians[this->numberOfGaussians - 1].getWeight());
        this->gaussians[this->numberOfGaussians - 1] = gaussian;
    }

    for (int i = 0; i < this->numberOfGaussians; i++) {
        this->gaussians[i].setWeight(this->gaussians[i].getWeight() / weightSum);
        this->gaussians[i].setWeightDistrRatio(this->gaussians[i].getWeight() / sqrt(this->gaussians[i].getVariance()));
    }

    // Sort the Gaussian components by their weightRatio
    sort(this->gaussians.begin(), this->gaussians.end(), [](Gaussian a, Gaussian b) {
        return a.getWeightDistrRatio() > b.getWeightDistrRatio();
    });

    return isBackgrond;
}



