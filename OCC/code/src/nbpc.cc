#include "../include/nbpc.h"

NBPC::NBPC() : GenericClassifier()
{
    this->t = 1.0;
}

arma::colvec NBPC::predict(arma::mat& X)
{
    arma::colvec y_pred(X.n_cols);
    for(size_t i = 0; i < X.n_cols; i++)
    {
        double p = 1;
        y_pred[i] = -1;
        for (size_t j = 0; j < X.n_rows; j++)
            p *= calculate_probability_distribution(this->mean(j), this->var(j), X(j, i));
        if (p >= this->t)
            y_pred[i] = 1;
    }
    return y_pred;
}

double NBPC::calculate_probability_distribution(double mean, double variance, double feature)
{
    if (variance == 0) return 0;    
    double a = (1 / std::sqrt(2*M_PI*variance));
    double b = (-1*std::pow((feature - mean),2))/(2*variance);
    double c = std::exp(b);
    double p = a*c;
    return p;
}


void NBPC::fit(arma::mat& X)
{
    this->mean.clear(); this->mean.zeros(X.n_rows);
    this->var.clear(); this->var.zeros(X.n_rows);
    for (size_t f=0; f < X.n_rows; f++)
    {
        this->mean(f) = arma::mean(X.row(f));
        this->var(f) = arma::var(X.row(f), 0);
    }

    double p;
    double temp;
    for (size_t i=0; i < X.n_cols; i++)
    {
        p = 1.0;
        for (size_t j=0; j<X.n_rows; j++)
        {
            temp = calculate_probability_distribution(this->mean(j), this->var(j), X.at(j, i));
            p *= temp;
        }
        if(p < this->t)
            this->t = p;
    }
}