#ifndef MODEL_H
#define MODEL_H
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include "ecoocc.h"
#include "heif.h"
#include "math_util.h"
#include "dcdistance_occ.h"
#include "oneclass_svm.h"
#include "nbpc.h"
#include "io.h"
#include "generic_classifier.h"
#include "tfidf_vectorizer.h"
#include <map>

const size_t FEATURES_CODE = 1;
const size_t WORD2VEC_CODE = 2;
const size_t BOW_CODE = 3;

const std::vector<std::string> FEATURES = {"uppercase", "exclamation", "has_exclamation", "question", "adj", "adv", "noun", "spell_errors", "lexical_size", "polarity", "number_sentences", "len_text", "words_per_sentence", "word2vec", "label"};
const std::map<std::string, int> feature_map = {{"uppercase", 0},
                                                {"exclamation", 1}, 
                                                {"has_exclamation", 2}, 
                                                {"question", 3},
                                                {"adj", 4},
                                                {"adv", 5},
                                                {"noun", 6},
                                                {"spell_errors", 7},
                                                {"lexical_size", 8},
                                                {"polarity", 9},
                                                {"number_sentences", 10},
                                                {"len_text", 11},
                                                {"words_per_sentence", 12},
                                                {"word2vec", 13},
                                                {"label", 14},
                                               };


class Model
{
    public:

        Model() {};

        double classify(std::string dataset_prefix, 
                        size_t n_folds, 
                        GenericClassifier* clf,
                        bool print_results=false,
                        size_t feature_set=FEATURES_CODE,
                        std::vector<std::string> features=FEATURES);

        double classify_bow(std::string dataset_prefix, 
                           size_t n_folds, 
                           GenericClassifier* clf, 
                           bool print_results=false);

        arma::mat classification_report(arma::colvec y_true, 
                                        arma::colvec y_pred, 
                                        short pos_label=1);

        void manual_gridsearch_dcdistance(arma::colvec thresholds, 
                                          std::string dataset_prefix, 
                                          size_t n_folds=10,
                                          size_t feature_set=FEATURES_CODE,
                                          std::vector<std::string> features=FEATURES);

        void manual_gridsearch_occsvm(std::vector<std::string> nu, 
                                      std::vector<std::string> kernel, 
                                      std::vector<std::string> degree, 
                                      std::string dataset_prefix, 
                                      size_t n_folds=10,
                                      size_t feature_set=FEATURES_CODE,
                                      std::vector<std::string> features=FEATURES);

        void manual_gridsearch_heif(std::vector<size_t> n_trees, 
                                    std::vector<size_t> sample_size, 
                                    std::vector<size_t> extension_level,
                                    std::string dataset_prefix, 
                                    size_t n_folds,
                                    size_t feature_set,
                                    bool random_picks=false,
                                    std::vector<std::string> features=FEATURES);
        
        arma::mat select_features(std::vector<std::string> features, arma::mat& X);

        private:
            arma::mat true_data_features;
            arma::mat fake_data_features;
            arma::mat true_data_word2vec;
            arma::mat fake_data_word2vec;
            std::vector<std::string> true_data_text;
            std::vector<std::string> fake_data_text;
};

#endif
