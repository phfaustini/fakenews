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

#ifndef HEIF_H
#define HEIF_H
#include "math_util.h"
#include "generic_classifier.h"
#include <vector>
#include <boost/random.hpp>
#include <string>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <memory>


class ITree
{
    /*A single tree in the forest that is build using a unique subsample.*/

    public:
        /**Public constructor.
         * 
         * @param X: Data present at the root node of this tree.
         * 
         * @param obj_idxs: indexes of objects present at the root node of this tree.
         * 
         * @param current_height: Depth of tree. current_height <= limit.
         * 
         * @param limit: Maximum depth a tree can reach before its creation is terminated.
         * 
         * @param extension_level: Extension level used in the splitting criteria.        
         * 
         * @param random_picks: Whether to select intercept point and normal vector randomly (true)
         *                      or computing hyperplane (false). If true, it is the same as IF and EIF.
        */
        ITree(arma::mat *X, 
              arma::uvec obj_idxs, 
              size_t current_height, 
              size_t limit, 
              size_t extension_level=0, 
              bool random_picks=false);

        ~ITree();

        /**
         * Computes the euclidean distance from object x to this tree's centroid.
         * 
         * @param x: a colvec object
         * 
         * @return: distance from x to this tree's centroid.
         */
        double distance_to_centroid(arma::colvec& x);

        /*Print tree*/
        std::string print(std::string space="");

        /*Getters and setters*/
        std::string get_ntype();
        size_t get_size();
        size_t get_current_height();
        size_t get_max_depth();
        arma::colvec get_normal();
        arma::colvec get_intercept();
        std::shared_ptr<ITree> get_left();
        std::shared_ptr<ITree> get_right();
        arma::colvec get_centroid();

    protected:
        size_t extension_level; // Extention level to be used in the creating splitting critera (default=0, same as IF).
        size_t current_height; // Depth of the tree to which the node belongs.
        size_t node_size; // Number of objects of the dataset present at the node.
        size_t dim; // Number of features.
        arma::colvec intercept_p; // Intercept point through which the hyperplane passes.
        arma::colvec normal; // Normal vector used to build the hyperplane that splits the data in the node.
        std::string ntype; // The type of the node: 'exNode', 'inNode'.  
        std::shared_ptr<ITree> right; // Left child node.
        std::shared_ptr<ITree> left; // Right child node.
        bool random_picks; // Whether to use IF and EIF methods (true), or HEIF (false, default).
        arma::colvec centroid; // Centroid of data (if leaf node)
        static boost::random::mt19937 gen;

    private:
        /**
         * Finds a orthogonal vector to the given vector x.
         * 
         * @param a: a column vector.
         * 
         * @return: a column vector perpendicular to x.
         */
        arma::colvec find_orthogonal_vector(arma::colvec x);

        /**
         * Finds the middle point between two points.
         * 
         * The middle point is simple the average of each
         * ith-element. 
         * 
         * Both params must have same size.
         * 
         * @param a: a column vector.
         * 
         * @param b: a column vector.
         * 
         * @return: the middle point between a and b.
         */
        arma::colvec find_middle_point(arma::colvec a, arma::colvec b);

        /**
         * Selects the rows (features) with the biggest variances.
         * The number of rows to be selected is according to 
         * extension level. These features will NOT be considered.
         * 
         * @param X: data (features, objects)
         * 
         * @return: list of feature indexes to be set to 0
         */
        arma::uvec biggest_variances_idxs(arma::mat& X);
        
        /**
         * Returns a random double within the specified range.
         * 
         * @param fMin: minimum possible value to be returned.
         * 
         * @param fMax: maximum possible value to be returned.
         * 
         * @return: a value between fMin and fMax.
         */
        double double_rand_range(double fMin, double fMax);

        /**
         * Returns a random size_t within the specified range.
         * 
         * @param iMin: minimum possible value to be returned.
         * 
         * @param iMax: maximum possible value to be returned.
         * 
         * @return: a value between iMin and iMax.
         */
        size_t sizet_rand_range(size_t iMin, size_t iMax);

};

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

class IForest
{
    /*iForest Class. It holds the data as well as the trained trees (iTree objects).*/

    public:
        /**
         * Public constructor.
         * 
         * @param X: Training data. Each column is an object and each row is a feature.
         * 
         * @param ntrees: Number of trees to be used. If HEIF method, must be at least 10.
         * 
         * @param sample_size: Number of objects to be used in creation of each tree. Must be smaller than |X|.
         * 
         * @param extension_level: Degree of freedom in choosing the hyperplanes for dividing up data. 
         *                         Must be smaller than the dimension n of the dataset.
         *                         If 0, is equivalent to standard Isolation Forest.
         * 
         * @param random_picks: Whether to select intercept point and normal vector randomly (true)
         *                      or computing hyperplane (false). If true, it is the same as IF and EIF.
         * 
         * @param hybrid: Whether to use Hybrid Isolation Forest (true) or not (false).
         * 
         * @param alpha: Percentage of the regular anomaly score to use. Ignored if hybrid = false
         */
        IForest(arma::mat& X, 
                size_t ntrees, 
                size_t sample_size,
                size_t extension_level=0, 
                bool random_picks=false, 
                bool hybrid=false, 
                double alpha=0.3);

        ~IForest();

        /**
         * This method makes sure the extension level provided by the user does not exceed the dimension of the data.
         *  
         * @return: void. Throws exception in the case of a violation.
         */
        void check_extension_level();

        /**
         * Computes anomaly scores for all data points in a dataset X_in 
         * 
         * @param X_in: Data to be scored. iForest.Trees are used for computing the depth reached in each tree by each data point.
         * 
         * @return: anomaly score for data points. 
         */
        arma::colvec anomaly_score(arma::mat& X_in);

        /**
         * @return the average depth of its trees.
         */
        double get_mean_depth();

        /**
         * Computes the mean euclidean distance from test objects to all trees' centroids.
         * 
         * @param X_in: Data to be scored. iForest.Trees are used for computing the distance to each object and tree's centroid.
         * 
         * @return: vector with mean distance from each x to trees' centroid.
         */
        arma::colvec sc(arma::mat& X_in);

        /*Getters */
        double get_min_anomaly_train();
        double get_max_anomaly_train();
        double get_min_sc();
        double get_max_sc();


    protected:
        /**
         * Computes the path length a given object x has
         * in tree t by traversing the point on the tree
         * until it reaches an external node.
         * 
         * @param x: a feature vector object.
         * 
         * @param t: a tree.
         * 
         * @param current_height: current height of the tree
         * 
         * @return: path length.
         */
        double path_length(arma::colvec x, std::shared_ptr<ITree> t, size_t current_height);

        /**Average path length of unsuccesful search in a binary search tree given n points 
         * 
         * @param n: Number of data points for the BST.
         * 
         * @return: average path length of unsuccesful search in a BST
        */
        double c_factor(size_t n);

    private:
        arma::mat X; // Data used for training.
        size_t nobjs; // Dataset size.
        std::vector<std::shared_ptr<ITree>> Trees; // A list of smart pointers to tree objects.
        size_t ntrees; // Number of trees to be used.
        size_t sample; //  Number of the objects to be used for tree creation.
        size_t limit; // Maximum height of a tree.
        size_t extension_level; // Extention level to be used in the creating splitting critera.
        double c; // Multiplicative factor used in computing the anomaly scores.
        bool random_picks; // Whether to use IF and EIF methods (true), or HEIF (false, default).
        double min_anomaly_train;
        double max_anomaly_train;
        double min_sc;
        double max_sc;
        bool hybrid; // Whether to use Hybrid Isolation Forest (true) or not (false).
        double alpha; // Percentage of the regular anomaly score to use. Ignored if hybrid = false
};



/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////


class HEIF  : public GenericClassifier
{
    public:
        /**
         * Public constructor.
         * 
         * @param ntrees: Number of trees to be used.
         * 
         * @param sample_size: Number of objects to be used in creation of each tree. Must be smaller than |X|.
         * 
         * @param extension_level: Degree of freedom in choosing the hyperplanes for dividing up data. 
         *                         Must be smaller than the dimension n of the dataset.
         *                         If 0, is equivalent to standard Isolation Forest.
         * 
         * @param random_picks: Whether to select intercept point and normal vector randomly (true)
         *                      or computing hyperplane (false). If true, it is the same as IF and EIF.
         * 
         * @param hybrid: Whether to use Hybrid Isolation Forest (true) or not (false).
         * 
         * @param alpha: Percentage of the regular anomaly score to use. Ignored if hybrid = false
         */
        HEIF(size_t ntrees, 
             size_t sample_size,
             size_t extension_level=0, 
             bool random_picks=false, 
             bool hybrid=false, 
             double alpha=0.3, 
             size_t random_state=42,
             bool debug=false);

        ~HEIF();

        std::string whoami() {return "HEIF.";}

        /**
         * Fit estimator.
         * 
         * @param X: The input samples  (n_features, n_samples).
         */
        void fit(arma::mat& X);

        /**
         * Computes anomaly scores for each object. The higher the score,
         * more likely an object is outlier.
         * 
         * @param X: The input samples  (n_features, n_samples).
         * 
         * @return: anomaly score for each sample, between 0 and 1.
         * 
         */
        arma::colvec score_samples(arma::mat& X);

        /**
         * Predict if a particular sample is an outlier or not.
         * 
         * @param X: The input samples  (n_features, n_samples).
         * 
         * @return: For each observation, +1 is inlier and -1 is outlier.
         */
        arma::colvec predict(arma::mat& X);

        /*Getters and setters*/
        void set_ntrees(size_t ntrees);
        void set_sample_size(size_t sample_size);
        void set_extension_level(size_t extension_level);

    protected:
        IForest *forest;
        size_t ntrees;
        size_t sample_size;
        size_t extension_level;
        bool random_picks;
        bool hybrid;
        double alpha;
        bool debug;
};

#endif
