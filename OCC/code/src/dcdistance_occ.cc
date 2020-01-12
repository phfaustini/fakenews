#include "../include/dcdistance_occ.h"


DCDistanceOCC::DCDistanceOCC(double t, const size_t metric) : GenericClassifier()
{
    this->t = t;
    this->metric = metric;
}

DCDistanceOCC::~DCDistanceOCC()
{
    distances.clear(); 
    class_vector.clear();
}


void DCDistanceOCC::fit(arma::mat& X)
{
    this->class_vector = arma::sum(X, 1);
    this->distances.reset();
    this->distances = arma::zeros<arma::colvec>(X.n_cols);
    for(size_t i = 0; i < X.n_cols; i++)
    {
        distances[i] = MathUtil::get_instance()->distance(this->class_vector, X.col(i), this->metric);
    }
    this->d = t * distances.max();
}


arma::colvec DCDistanceOCC::predict(arma::mat& X)
{
    arma::colvec y_pred(X.n_cols);
    for(size_t i = 0; i < X.n_cols; i++)
    {
        if (MathUtil::get_instance()->distance(this->class_vector, X.col(i), this->metric) <= this->d)
            y_pred[i] = 1;
        else 
            y_pred[i] = -1;
    }
    return y_pred;
}
