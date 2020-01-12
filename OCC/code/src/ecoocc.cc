#include "../include/ecoocc.h"

EcoOCC::EcoOCC(const size_t maxIterations, const size_t metric) : GenericClassifier()
{
    this->maxIterations = maxIterations;
    this->metric = metric;
    mlpack::math::RandomSeed(42);
}

EcoOCC::~EcoOCC()
{
    radii.clear();
    centroids_coordinates.clear();
}

double EcoOCC::silhouette(arma::mat X, arma::Row<size_t> assignments, arma::mat centroids)
{
    double a = 0;
    double silhouette_total = 0;
    for (size_t i = 0; i < X.n_cols; i++)
    {
        std::vector<double> clusters_distances;
        for (size_t j = 0; j < centroids.n_cols; j++)
        {
            if (j == assignments(i)) // Mean dst to other elements from same cluster.
            {
                a = MathUtil::get_instance()->distance(X.col(i), centroids.col(j), this->metric);
            }
            else // Distances to elements from other clusters.
            {
                clusters_distances.push_back(MathUtil::get_instance()->distance(X.col(i), centroids.col(j), this->metric));
            }
        }
        auto b = std::min_element(std::begin(clusters_distances), std::end(clusters_distances));
        if (a != 0) // If cluster has 1 element, then a is 0, and silhouete is also set to 0.
        {
            if (a < *b)
                silhouette_total += (1 - (a / *b));
            else 
                if (a > *b) silhouette_total += ((*b / a) - 1);
        }
    }
    return silhouette_total / X.n_cols;
}

void EcoOCC::fit_cosine(arma::mat& X)
{
    arma::Row<size_t> assignments;
    arma::mat centroids;
    size_t limit = (size_t)sqrt(X.n_cols); // Each column is an object.
    size_t best_k = 2; double best_silhouette = -2; double current_silhouette_score;
    for (size_t clusters = 2; clusters <= limit; clusters++)
    {
        mlpack::kmeans::KMeans<MyCosineDistance> k (this->maxIterations);
        assignments.clear();
        centroids.clear();
        k.Cluster(X, clusters, assignments, centroids);
        current_silhouette_score = silhouette(X, assignments, centroids);
        if (current_silhouette_score > best_silhouette)
        {
            best_silhouette = current_silhouette_score;
            best_k = clusters;
        }
    }
    std::string f = "results/best_k.txt";std::string s;s.append(std::to_string(best_k));s.append("\n");IO::get_instance()->append_file(f, s);
    mlpack::kmeans::KMeans<MyCosineDistance> k (this->maxIterations);
    assignments.clear();
    centroids.clear();
    k.Cluster(X, best_k, assignments, centroids);
    this->radii = std::vector<double>(best_k, 0.0);
    this->centroids_coordinates = centroids;
    double dst;
    for (size_t i = 0; i < X.n_cols; i++)
    {
        dst = MathUtil::get_instance()->distance(X.col(i), this->centroids_coordinates.col(assignments.at(i)), this->metric);
        if (dst > this->radii[assignments.at(i)])
            this->radii[assignments.at(i)] = dst;
    }
}

void EcoOCC::fit_euclidean(arma::mat& X)
{
    arma::Row<size_t> assignments;
    arma::mat centroids;
    size_t limit = (size_t)sqrt(X.n_cols); // Each column is an object.
    size_t best_k = 2; double best_silhouette = -2; double current_silhouette_score;
    for (size_t clusters = 2; clusters <= limit; clusters++)
    {
        mlpack::kmeans::KMeans<> k (this->maxIterations);
        assignments.clear();
        centroids.clear();
        k.Cluster(X, clusters, assignments, centroids);
        current_silhouette_score = silhouette(X, assignments, centroids);
        if (current_silhouette_score > best_silhouette)
        {
            best_silhouette = current_silhouette_score;
            best_k = clusters;
        }
    }
    std::string f = "results/best_k.txt";std::string s;s.append(std::to_string(best_k));s.append("\n");IO::get_instance()->append_file(f, s);
    mlpack::kmeans::KMeans<> k (this->maxIterations);
    assignments.clear();
    centroids.clear();
    k.Cluster(X, best_k, assignments, centroids);
    this->radii = std::vector<double>(best_k, 0.0);
    this->centroids_coordinates = centroids;
    double dst;
    for (size_t i = 0; i < X.n_cols; i++)
    {
        dst = MathUtil::get_instance()->distance(X.col(i), this->centroids_coordinates.col(assignments.at(i)), this->metric);
        if (dst > this->radii[assignments.at(i)])
            this->radii[assignments.at(i)] = dst;
    }
}

void EcoOCC::fit(arma::mat& X)
{
    if (this->metric == EUCLIDEAN)
        fit_euclidean(X);
    else //(this->metric == COSINE)
        fit_cosine(X);
}

arma::colvec EcoOCC::predict(arma::mat& X)
{
    arma::colvec y_pred(X.n_cols);
    for (size_t i = 0; i < X.n_cols; i++)
        y_pred.at(i) = this->predict_instance(X.col(i));
    return y_pred;
}

int EcoOCC::predict_instance(arma::colvec x)
{
    for (size_t i = 0; i < this->centroids_coordinates.n_cols; i++)
       if (MathUtil::get_instance()->distance(x, this->centroids_coordinates.col(i), this->metric) <= this->radii[i])
        return 1;
    return -1;
}