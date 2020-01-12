#include "../include/math_util.h"

MathUtil* MathUtil::pSingleton = NULL;

MathUtil::MathUtil()
{
   // do init stuff
}

MathUtil* MathUtil::get_instance()
{
	if (pSingleton == NULL) {
		pSingleton = new MathUtil();
	}
	return pSingleton;
}


arma::mat MathUtil::drop_last_row(arma::mat& matrix)
{
    return matrix.rows(0, matrix.n_rows-2);
}

arma::colvec MathUtil::get_last_row(arma::mat& matrix)
{
    arma::mat subm = matrix.submat(matrix.n_rows-1, 0, matrix.n_rows-1, matrix.n_cols-1);
    return arma::conv_to<arma::colvec>::from(subm);
}

double MathUtil::cosine_distance(arma::colvec A, arma::colvec B)
{
    return 1 - mlpack::kernel::CosineDistance::Evaluate(A, B); // Despite the name, mlpack computes cosine similarity.
}

double MathUtil::euclidean_distance(arma::colvec A, arma::colvec B)
{
    double s = 0;
    for (size_t i = 0; i < A.n_rows; i++)
    {
        s += pow((A(i) - B(i)), 2);
    }
    return sqrt(s);
}

double MathUtil::distance(arma::colvec A, arma::colvec B, size_t distance)
{
    if (distance == EUCLIDEAN)
    {
        return this->euclidean_distance(A, B);
    }
    else //if (distance == COSINE)
    {
        return this->cosine_distance(A, B);
    }
}