#ifndef DCDISTANCE_OCC_H
#define DCDISTANCE_OCC_H
#include "math_util.h"
#include "generic_classifier.h"
#include <cmath>

class DCDistanceOCC : public GenericClassifier
{
    public:
        DCDistanceOCC(double t, const size_t metric=EUCLIDEAN);
        ~DCDistanceOCC();
        void fit(arma::mat& X);
        arma::colvec predict(arma::mat& X);
        double getT() {return t;}
        std::string whoami() {return "DCDistanecOCC.";}

        
    protected:
        double t;
        double d;
        arma::colvec distances;
        arma::colvec class_vector;
        size_t metric;

    private:
        
};

#endif