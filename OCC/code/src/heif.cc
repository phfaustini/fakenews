/*
Copyright <2019> <Pedro Faustini>

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
THE SOFTWARE.
 */

#include "../include/heif.h"


/*IFOREST*/
IForest::IForest(arma::mat& X, 
                 size_t ntrees, 
                 size_t sample_size,
                 size_t extension_level, 
                 bool random_picks, 
                 bool hybrid, 
                 double alpha)
{
    this->ntrees = ntrees;
    this->X = X;
    this->nobjs = X.n_cols; // Each object is a column.
    this->sample = sample_size;
    this->Trees.clear();
    this->extension_level = extension_level;
    this->random_picks = random_picks;
    this->hybrid = hybrid;
    this->alpha = alpha;
    this->check_extension_level();
    this->c = this->c_factor(this->sample);
    this->limit = (int)(std::ceil(std::log2(this->sample)));
    if(this->random_picks) // EIF or IF
    {
        for (size_t i = 0; i < this->ntrees; i++)
        {
            arma::colvec temp_obj_idxs = arma::linspace(0, this->nobjs-1, this->nobjs);
            arma::uvec obj_idxs = arma::find(temp_obj_idxs);
            obj_idxs = arma::shuffle(obj_idxs);
            obj_idxs.resize(this->sample);
            this->Trees.push_back(std::shared_ptr<ITree>(new ITree(&X, obj_idxs, 0, this->limit, this->extension_level, this->random_picks)));
        }
    } // HEIF
    else
    {
        /**
         * Self adjustable.
         * 
         * First, it creates 10 trees with an absurd height limit
         * and checks their average height.
         * 
         * The limit that will be used will be the average height.
         */
        if(this->ntrees < 10)
            throw std::invalid_argument("HEIF: ntrees must be >= 10"); 
        for (size_t i = 0; i < 10; i++)
        {
            arma::colvec temp_obj_idxs = arma::linspace(0, this->nobjs-1, this->nobjs);
            arma::uvec obj_idxs = arma::find(temp_obj_idxs);
            obj_idxs = arma::shuffle(obj_idxs);
            obj_idxs.resize(this->sample);
            this->Trees.push_back(std::shared_ptr<ITree>(new ITree(&X, obj_idxs, 0, 1000, this->extension_level, this->random_picks)));
        }
        this->limit = get_mean_depth();
        this->Trees.clear();
        for (size_t i = 0; i < this->ntrees; i++)
        {
            arma::colvec temp_obj_idxs = arma::linspace(0, this->nobjs-1, this->nobjs);
            arma::uvec obj_idxs = arma::find(temp_obj_idxs);
            obj_idxs = arma::shuffle(obj_idxs);
            obj_idxs.resize(this->sample);
            this->Trees.push_back(std::shared_ptr<ITree>(new ITree(&X, obj_idxs, 0, this->limit, this->extension_level, this->random_picks)));
        }
    }
    if(this->hybrid) // Hybrid works with either methods
    {
        arma::colvec anomaly_scores = anomaly_score(this->X);
        arma::colvec sc_scores = sc(this->X);
        this->min_anomaly_train = anomaly_scores.min();
        this->max_anomaly_train = anomaly_scores.max();
        this->min_sc = sc_scores.min();
        this->max_sc = sc_scores.max();
    }
}

IForest::~IForest()
{

}

void IForest::check_extension_level()
{
    size_t dim = this->X.n_rows; // Each row is a feature.
    if (this->extension_level > dim - 1)
       throw std::invalid_argument("Extension level can't be higher than dim - 1");
}

arma::colvec IForest::anomaly_score(arma::mat& X_in)
{
    if (X_in.empty())
        X_in = this->X;
    arma::colvec S = arma::zeros(X_in.n_cols);
    for (size_t i = 0; i < X_in.n_cols; i++)
    {
        double h_temp = 0;
        for (size_t j = 0; j < this->ntrees; j++)
            h_temp += this->path_length(X_in.col(i), this->Trees[j], 0);
        double Eh = h_temp / this->ntrees;
        double as = std::pow(2.0, -Eh / this->c);
        S.row(i) = as; // Anomaly Score
    }
    return S;
}

double IForest::get_mean_depth()
{
    double avg_height = 0;
    for (size_t i = 0; i < this->Trees.size(); i++)
        avg_height += this->Trees[i]->get_max_depth();
    avg_height /= this->Trees.size();
    return std::round(avg_height);
}

arma::colvec IForest::sc(arma::mat& X_in)
{
    if (X_in.empty())
        X_in = this->X;
    arma::colvec S = arma::zeros(X_in.n_cols);
    for (size_t i = 0; i < X_in.n_cols; i++)
    {
        arma::colvec x = X_in.col(i);
        double s = 0;
        size_t n_trees = this->ntrees;
        for (size_t j = 0; j < this->ntrees; j++)
        {
            double dst = this->Trees[j]->distance_to_centroid(x);
            if(dst >= 0) // Negative distance means no centroid at that leaf (empty leaf)
                s += dst;
            else        // Ignore empty leaf
                n_trees--;
        }
        s /= n_trees;
        S.row(i) = s;
    }
    return S;
}

/*Getters */
double IForest::get_min_anomaly_train(){return this->min_anomaly_train;}
double IForest::get_max_anomaly_train(){return this->max_anomaly_train;}
double IForest::get_min_sc(){return this->min_sc;}
double IForest::get_max_sc(){return this->max_sc;}


/*PRIVATE*/
double IForest::path_length(arma::colvec x, std::shared_ptr<ITree> t, size_t current_height)
{
    if (t->get_ntype() == "exNode")
    {
        if(t->get_size() <= 1)
            return current_height;
        return current_height + this->c_factor(t->get_size());
    }
    arma::colvec normal = t->get_normal();
    arma::colvec p = t->get_intercept();
    if (arma::dot((x - p), normal) < 0)
        return this->path_length(x, t->get_left(), current_height+1);
    return this->path_length(x, t->get_right(), current_height+1);
}

double IForest::c_factor(size_t n)
{
    return 2.0*(std::log(n-1)+0.5772156649) - (2.0*(n-1.0)/(n*1.0));
}



/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////



/*ITREE*/
ITree::ITree(arma::mat *X, arma::uvec obj_idxs, size_t current_height, size_t limit, size_t extension_level, bool random_picks)
{
    this->current_height = current_height;
    this->extension_level = extension_level;
    this->dim = X->n_rows;
    this->node_size = obj_idxs.size();
    this->random_picks = random_picks;

    if (this->current_height >= limit || this->node_size <= 1)
    {
        this->ntype = "exNode";
        this->left.reset();
        this->right.reset();
        if (this->node_size >= 1)
            this->centroid = arma::mean(X->cols(obj_idxs), 1);
    }
    else
    {
        this->ntype = "inNode";

        if(this->random_picks) // IF or EIF methods
        {
            this->normal = arma::randn<arma::vec>(this->dim); //4

            arma::colvec mins = arma::min(X->cols(obj_idxs), 1); //5
            arma::colvec maxs = arma::max(X->cols(obj_idxs), 1);
            this->intercept_p = arma::zeros(this->dim);
            for (size_t i = 0; i < this->dim; i++)
            {
                this->intercept_p.at(i) = this->double_rand_range(mins.at(i), maxs.at(i));
            }
            
            //Pick the indexes for which the normal vector elements should be set to zero acccording to the extension level.
            arma::colvec feature_idxs = arma::linspace(0, this->dim-1, this->dim);
            feature_idxs = arma::shuffle(feature_idxs);
            feature_idxs.resize(this->dim - this->extension_level - 1);
            for (size_t i = 0; i < feature_idxs.size(); i++)
                this->normal.at(feature_idxs.at(i)) = 0; //6
        }
        else // HEIF method
        {
            arma::mat centroids;
            bool status = arma::kmeans(centroids, X->cols(obj_idxs), 2, arma::static_spread, 50, false);
            if (!status)
                throw std::runtime_error("clustering failed");
            arma::colvec a = centroids.col(0);
            arma::colvec b = centroids.col(1);
            this->normal = a-b;//find_orthogonal_vector(a - b); //4
            this->intercept_p = find_middle_point(a, b); //5

            //Pick the indexes for which the normal vector elements should be set to zero acccording to the extension level.
            arma::mat xp = X->cols(obj_idxs);
            arma::uvec feature_idxs = biggest_variances_idxs(xp);
            xp.clear();
            for (size_t i = 0; i < feature_idxs.size(); i++)
                this->normal.at(feature_idxs.at(i)) = 0; //6
        }

        arma::uvec Xr(obj_idxs.size());
        arma::uvec Xl(obj_idxs.size());
        size_t counterL = 0;
        size_t counterR = 0;
        for (size_t i = 0; i < obj_idxs.size(); i++)
        {
            arma::colvec obj = X->col(obj_idxs.at(i));
            double v = arma::dot((obj - this->intercept_p), this->normal);
            if(v <= 0)
                Xl.at(counterL++) = obj_idxs.at(i); //7
            else
                Xr.at(counterR++) = obj_idxs.at(i); //8
        }
        Xr.resize(counterR);
        Xl.resize(counterL);
        this->right = std::shared_ptr<ITree>(new ITree(X, Xr, this->current_height+1, limit, this->extension_level, this->random_picks));
        this->left = std::shared_ptr<ITree>(new ITree(X, Xl, this->current_height+1, limit, this->extension_level, this->random_picks));
    }
}

ITree::~ITree()
{
    
}

double ITree::distance_to_centroid(arma::colvec& x)
{
    if(this->ntype == "inNode") // Recursively traverses tree until reaches a leaf
    {
        double v = arma::dot((x - this->intercept_p), this->normal);
        if(v <= 0)
            return this->left->distance_to_centroid(x);
        else
            return this->right->distance_to_centroid(x);
    }
    // It is in a leaf node now
    if(this->centroid.is_empty())
        return -1;
    double s = 0;
    for (size_t i = 0; i < x.n_rows; i++)
    {
        s += std::pow((x(i) - this->centroid(i)), 2);
    }
    return std::sqrt(s);
}

std::string ITree::print(std::string space)
{
    std::string s = space + "e = " + std::to_string(this->current_height) + ", Size = " + std::to_string(this->node_size) + " " + this->ntype + "\n";
    if(this->left.get() != nullptr) s+= space + "Xl: " + this->left->print(space+"    ") + "\n";
    if(this->right.get() != nullptr) s+= space + "Xr: " + this->right->print(space+"    ") + "\n";
    return s;
}

std::string ITree::get_ntype(){return this->ntype;}
size_t ITree::get_size(){return this->node_size;}
size_t ITree::get_current_height(){return this->current_height;}
arma::colvec ITree::get_normal(){return this->normal;}
arma::colvec ITree::get_intercept(){return this->intercept_p;}
std::shared_ptr<ITree> ITree::get_left(){return this->left;}
std::shared_ptr<ITree> ITree::get_right(){return this->right;}
arma::colvec ITree::get_centroid(){return this->centroid;}
size_t ITree::get_max_depth()
{
    if (this->ntype == "exNode")
        return 0;
    return 1 + std::max(this->left->get_max_depth(), this->right->get_max_depth());
}

boost::random::mt19937 ITree::gen;

/*PRIVATE*/
arma::colvec ITree::find_orthogonal_vector(arma::colvec x)
{
    // https://math.stackexchange.com/questions/133177/finding-a-unit-vector-perpendicular-to-another-vector
    arma::colvec w = arma::randu<arma::vec>(x.n_rows);
    return w - x * (arma::dot(x, w) / arma::dot(x,x));
}

arma::colvec ITree::find_middle_point(arma::colvec a, arma::colvec b)
{
    arma::colvec middle = arma::zeros(a.n_rows);
    for (size_t i = 0; i < a.n_rows; i++)
        middle.at(i) = (a.at(i) + b.at(i)) / 2;
    return middle;
}

arma::uvec ITree::biggest_variances_idxs(arma::mat& X)
{
    size_t top_features = this->dim - (this->extension_level + 1);
    arma::colvec vars = arma::var(X, 0, 1);
    arma::uvec feature_idxs = arma::stable_sort_index(vars, "descend");
    feature_idxs.resize(top_features);    
    return feature_idxs;
}

double ITree::double_rand_range(double fMin, double fMax)
{
    if(fMin == fMax)
        return fMin;
    boost::random::uniform_real_distribution<> dist(fMin, fMax);
    return dist(ITree::gen);
}

size_t ITree::sizet_rand_range(size_t iMin, size_t iMax)
{
    if(iMin == iMax)
        return iMin;
    boost::random::uniform_int_distribution<> dist(iMin, iMax);
    return dist(ITree::gen);
}
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////


HEIF::HEIF(size_t ntrees, 
           size_t sample_size,
           size_t extension_level, 
           bool random_picks,
           bool hybrid,
           double alpha,
           size_t random_state,
           bool debug) : GenericClassifier()
{
    arma::arma_rng::set_seed(random_state);
    std::srand(random_state);
    this->forest = nullptr;
    this->ntrees = ntrees;
    this->sample_size = sample_size;
    this->extension_level = extension_level;
    this->random_picks = random_picks;
    this->hybrid = hybrid;
    this->alpha = alpha; 
    this->debug = debug;
}

HEIF::~HEIF()
{
    if(this->debug) std::cout << "Average depth: " << this->forest->get_mean_depth() << std::endl;
}

void HEIF::fit(arma::mat& X)
{
    delete this->forest;
    this->forest = new IForest(X, this->ntrees, this->sample_size, this->extension_level, this->random_picks, this->hybrid, this->alpha);
}

arma::colvec HEIF::score_samples(arma::mat& X)
{
    arma::colvec sn = this->forest->anomaly_score(X);
    if(this->hybrid)
    {
        arma::colvec sc = this->forest->sc(X);
        sc = (sc + this->forest->get_min_sc()) / (this->forest->get_max_sc() - this->forest->get_min_sc());
        sn = (sn + this->forest->get_min_anomaly_train()) / (this->forest->get_max_anomaly_train() - this->forest->get_min_anomaly_train());
        arma::colvec shif1 = arma::zeros(X.n_cols);
        shif1 = this->alpha * sn + (1 - this->alpha) * sc;
        return shif1;
    }
    return sn;
}

arma::colvec HEIF::predict(arma::mat& X)
{
    arma::colvec scores = this->score_samples(X);
    arma::colvec y_pred = arma::zeros(X.n_cols);
    for (size_t i = 0; i < X.n_cols; i++)
        y_pred.at(i) = (scores.at(i) <= 0.5) ? 1 : -1;
    return y_pred;
}

void HEIF::set_ntrees(size_t ntrees){this->ntrees = ntrees;}
void HEIF::set_sample_size(size_t sample_size){this->sample_size = sample_size;}
void HEIF::set_extension_level(size_t extension_level){this->extension_level = extension_level;}