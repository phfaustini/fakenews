#ifndef ECOOCC_H
#define ECOOCC_H
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/core/math/random.hpp>
#include <vector>
#include "math_util.h"
#include "io.h"
#include "generic_classifier.h"
#include "cosine_distance.h"

//https://stackoverflow.com/questions/40501935/mlpack-error-while-building
class EcoOCC : public GenericClassifier
{
    public:
        EcoOCC(const size_t maxIterations, const size_t metric=EUCLIDEAN);
        ~EcoOCC();
        void fit(arma::mat& X);
        arma::colvec predict(arma::mat& X);
        std::string whoami() {return "EcoOCC.";}
        

    private:
        int predict_instance(arma::colvec x);
        double silhouette(arma::mat X, arma::Row<size_t> assignments, arma::mat centroids);
        std::vector<double> radii;
        arma::mat centroids_coordinates;
        size_t maxIterations;
        size_t metric;
        void fit_cosine(arma::mat& X);
        void fit_euclidean(arma::mat& X);
};


#endif