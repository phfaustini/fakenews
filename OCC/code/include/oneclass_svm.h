#ifndef ONECLASS_SVM_H
#define ONECLASS_SVM_H
#include "math_util.h"
#include "io.h"
#include "generic_classifier.h"
#include <cstdlib>

const short KERNEL_LINEAR  = 0;
const short KERNEL_POLY    = 1;
const short KERNEL_RBF     = 2;
const short KERNEL_SIGMOID = 3;

class OCCSVM : public GenericClassifier
{
    public:
        OCCSVM(std::string kernel, std::string degree, std::string nu);
        ~OCCSVM(){};
        void fit(arma::mat& X);
        arma::colvec predict(arma::mat& X);
        std::string getSVMType() {return svm_type;}
        std::string getKernel() {return kernel;}
        std::string getDegree() {return degree;}
        std::string getNu() {return nu;}
        std::string whoami() {return "OCCSVM.";}

    protected:
        bool convert_arma_to_txt(arma::mat& X, std::string& target_filename);

    private:
        std::string svm_type;
        std::string kernel;
        std::string degree;
        std::string nu;
};

#endif