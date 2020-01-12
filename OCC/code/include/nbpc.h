#ifndef NBPC_H
#define NBPC_H
#include <cmath>
#include "math_util.h"
#include "generic_classifier.h"

class NBPC : public GenericClassifier
{
    public:
        NBPC();
        ~NBPC(){};
        void fit(arma::mat& X);
        arma::colvec predict(arma::mat& X);
        std::string whoami() {return "NBPC.";}

        
    protected:
        double calculate_probability_distribution(double mean, double variance, double feature);

    private:
        double t;
        arma::colvec mean;
        arma::colvec var;
        
};

#endif
