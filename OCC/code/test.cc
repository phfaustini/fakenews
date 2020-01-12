// g++ test.cc -larmadillo -lmlpack -fopenmp
#include <cstdlib>
#include <iostream>
#include <mlpack/core/util/version.hpp>
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include <set>
#include <utility>
#include <string>

arma::mat classification_report(arma::colvec y_true, arma::colvec y_pred, short pos_label)
{
    double tp = 0;
    double fp = 0;
    double fn = 0;
    double correct = 0;
    for (size_t i = 0; i < y_true.n_rows; i++)
    {
        if (y_true(i) == y_pred(i))
            correct++;
        if (y_true(i) == y_pred(i) && y_pred(i) == pos_label)
            tp++;
        else if (y_true(i) != y_pred(i) && y_pred(i) == pos_label)
            fp++;
        else if (y_true(i) != y_pred(i) && y_pred(i) != pos_label)
            fn++;
    }
    double accuracy = correct / y_true.n_rows;
    double precision = (tp + fp > 0)            ?    tp / (tp + fp) : 0;
    double recall    = (tp + fn > 0)            ?    tp / (tp + fn) : 0;
    double fscore    = (precision + recall > 0) ?    2 * ( (precision*recall) / (precision + recall) ) : 0;
    arma::mat a_p_r_f(1, 4);
    a_p_r_f(0, 0) = accuracy*100; a_p_r_f(0, 1) = precision*100; a_p_r_f(0, 2) = recall*100; a_p_r_f(0, 3) = fscore*100;
    return a_p_r_f;
}

int main()
{
    std::cout << mlpack::util::GetVersion() << std::endl;
    arma::mat M = arma::mat(4, 2);
    arma::colvec a = {3.1,  3.4,  9.2, -8.2};
    arma::colvec b = {-0.3,  2.2,  4. ,  1.1};
    M.col(0) = a;
    M.col(1) = b;
    M = arma::normalise(M, 2, 0);
    M.print("M");
    std::cout << std::endl;

    arma::colvec y_true = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    arma::colvec y_pred = {-1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    arma::mat prf = classification_report(y_true, y_pred, 1);
    prf.print();

    arma::colvec y_true2 = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    arma::colvec y_pred2 = { 1,-1, 1, 1, 1, 1, 1,-1, 1,-1,-1,-1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    arma::mat prf2 = classification_report(y_true2, y_pred2, 1);
    prf2.print();
    
    return 0;
}