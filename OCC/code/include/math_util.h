#ifndef MATH_UTIL_H
#define MATH_UTIL_H
#include <mlpack/core/kernels/cosine_distance.hpp>
#include <armadillo>
#include <cmath>

const size_t COSINE = 1;
const size_t EUCLIDEAN = 2;

class MathUtil {
public:
   static MathUtil* get_instance();
   arma::mat drop_last_row(arma::mat& matrix);
   arma::colvec get_last_row(arma::mat& matrix);
   double euclidean_distance(arma::colvec A, arma::colvec B);
   double cosine_distance(arma::colvec A, arma::colvec B);
   double distance(arma::colvec A, arma::colvec B, size_t distance=EUCLIDEAN);
   

private:
   MathUtil();
   static MathUtil* pSingleton;		// singleton instance
};

#endif
