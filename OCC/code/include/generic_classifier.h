#ifndef GENERIC_CLASSIFIER_H
#define GENERIC_CLASSIFIER_H
#include <string>
#include <armadillo>

class GenericClassifier
{
    public:
        GenericClassifier() {};
        virtual ~GenericClassifier(){};

        virtual void fit(arma::mat&) {};
        virtual arma::colvec predict(arma::mat&) {arma::colvec c; return c;};
        virtual std::string whoami() {return "Generic.";}
};

#endif