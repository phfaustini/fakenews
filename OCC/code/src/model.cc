#include "../include/model.h"

arma::mat Model::classification_report(arma::colvec y_true, arma::colvec y_pred, short pos_label)
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


double Model::classify_bow(std::string dataset_prefix, 
                           size_t n_folds, 
                           GenericClassifier* clf, 
                           bool print_results)
{
    mlpack::data::StandardScaler scaler;
    arma::mat accuracy_precision_recall_fscore;

    /*Load datasets if they have not been loaded yet.*/
    std::string filename_false = dataset_prefix + "text_False.csv";
    std::string filename_true = dataset_prefix + "text_True.csv";
    if (this->fake_data_text.empty()) this->fake_data_text = IO::get_instance()->load_text_dataset(filename_false);
    if (this->true_data_text.empty()) this->true_data_text = IO::get_instance()->load_text_dataset(filename_true); 
    
    /*Splitting objects into n folds.*/
    size_t k;
    std::vector<std::vector<std::string>> fake_folds(n_folds);
    for (size_t i = 0; i < this->fake_data_text.size(); i++)
    {
        k = i % n_folds;
        fake_folds[k].push_back(this->fake_data_text[i]);
    }
    std::vector<std::vector<std::string>> true_folds(n_folds);
    for (size_t i = 0; i < this->true_data_text.size(); i++)
    {
        k = i % n_folds;
        true_folds[k].push_back(this->true_data_text[i]);
    }

    /*Declaring necessary data structures. */
    std::vector<std::string> train_data;
    std::vector<std::string> test_false_data;
    std::vector<std::string> test_true_data;
    std::vector<std::string> test_data;
    arma::mat a_p_r_f;
    arma::colvec y_pred;

    for (size_t i = 0; i < n_folds; i++)
    {
        /*We select i-fold for testing. The others are used for training. */
        test_false_data.clear();
        test_false_data = fake_folds[i];
        train_data.clear();
        for (size_t k = 0; k < n_folds; k++)
            if (k != i)
                train_data.insert(std::end(train_data), std::begin(fake_folds[k]), std::end(fake_folds[k]));

        /*Preprocessing data and fitting model.*/
        //binary, lowercase, use_idf, max_features, norm, sublinear_tf
        TfIdfVectorizer vectoriser(false, true, true, 14, "0", false);
        arma::mat tfidf = vectoriser.fit_transform(train_data);
        scaler.Fit(tfidf);
        arma::mat train_data_transformed;
        scaler.Transform(tfidf, train_data_transformed);
        clf->fit(train_data_transformed);
        
        for (size_t j = 0; j < n_folds; j++)
        {
            test_true_data.clear();
            test_true_data = true_folds[j]; // Selecting one fold for testing.
            if (dataset_prefix == "datasets/WhatsApp/whats_br/Structured/")
                test_true_data = this->true_data_text;

            /*Set labels for y_pred*/
            std::vector<double> t; for (size_t z = 0; z < test_true_data.size(); z++) t.push_back(-1.0);
            std::vector<double> f; for (size_t z = 0; z < test_false_data.size(); z++) f.push_back(1.0);
            std::vector<double> labels; labels.insert(std::end(labels), std::begin(t), std::end(t)); labels.insert(std::end(labels), std::begin(f), std::end(f));
            arma::colvec y_true(labels);

            /*The two selected folders (one with true and other with fake data) are merged in one data structure*/
            test_data.clear();
            test_data.insert(std::end(test_data), std::begin(test_true_data), std::end(test_true_data));
            test_data.insert(std::end(test_data), std::begin(test_false_data), std::end(test_false_data));
            tfidf = vectoriser.transform(test_data);
            arma::mat test_data_transformed;
            scaler.Transform(tfidf, test_data_transformed);
            y_pred = clf->predict(test_data_transformed);
            a_p_r_f = classification_report(y_true, y_pred);
            accuracy_precision_recall_fscore = arma::join_vert(accuracy_precision_recall_fscore, a_p_r_f);
            if (dataset_prefix == "datasets/WhatsApp/whats_br/Structured/") break;
        }
    }
    arma::mat a_p_r_f_mean = arma::mean(accuracy_precision_recall_fscore, 0); // 0 = dim (mean of each column).
    arma::mat a_p_r_f_std = arma::stddev(accuracy_precision_recall_fscore, 0, 0); // 0s: Norm type, dim
    if (print_results)
    {
        std::cout << "Accuracy  = " << round(a_p_r_f_mean(0)) << "% \u00b1" << round(a_p_r_f_std(0)) << "%" << std::endl;
        std::cout << "Precision = " << round(a_p_r_f_mean(1)) << "% \u00b1" << round(a_p_r_f_std(1)) << "%" << std::endl;
        std::cout << "Recall    = " << round(a_p_r_f_mean(2)) << "% \u00b1" << round(a_p_r_f_std(2)) << "%" << std::endl;
        std::cout << "FScore    = " << round(a_p_r_f_mean(3)) << "% \u00b1" << round(a_p_r_f_std(3)) << "%" << std::endl;
    }
    delete clf;
    return a_p_r_f_mean(3);
}


arma::mat Model::select_features(std::vector<std::string> features, arma::mat& X)
{
    arma::mat F;
    int feature_index;
    for (size_t i = 0; i < features.size(); i++)
    {
        feature_index = feature_map.at(features[i]);
        F.insert_rows(i, X.row(feature_index));
    }
    return F;
}

double Model::classify(std::string dataset_prefix, 
                       size_t n_folds, 
                       GenericClassifier* clf, 
                       bool print_results, 
                       size_t feature_set, 
                       std::vector<std::string> features)
{
    mlpack::data::StandardScaler scaler;
    arma::mat accuracy_precision_recall_fscore;

    /*Load datasets if they have not been loaded yet.*/
    arma::mat fake_data;
    arma::mat true_data;
    std::string filename_false;
    std::string filename_true;
    switch (feature_set)
    {
        case FEATURES_CODE:
            filename_false = dataset_prefix + "features_False.csv";
            filename_true = dataset_prefix + "features_True.csv";
            if (this->fake_data_features.n_cols == 0) this->fake_data_features = IO::get_instance()->load_generic_dataset(filename_false); 
            if (this->true_data_features.n_cols == 0) this->true_data_features = IO::get_instance()->load_generic_dataset(filename_true);
            fake_data = select_features(features, this->fake_data_features);
            true_data = select_features(features, this->true_data_features);            
            break;
        case WORD2VEC_CODE:
            filename_false = dataset_prefix + "word2vec_False.csv";
            filename_true = dataset_prefix + "word2vec_True.csv";
            if (this->fake_data_word2vec.n_cols == 0) this->fake_data_word2vec = IO::get_instance()->load_generic_dataset(filename_false); 
            if (this->true_data_word2vec.n_cols == 0) this->true_data_word2vec = IO::get_instance()->load_generic_dataset(filename_true);
            fake_data = this->fake_data_word2vec;
            true_data = this->true_data_word2vec;
        default:
            break;
    }

    /*Splitting objects into n folds.*/
    size_t k;
    std::vector<arma::mat> fake_folds(n_folds);
    for (size_t i = 0; i < fake_data.n_cols; i++)
    {
        k = i % n_folds;
        fake_folds[k].insert_cols(fake_folds[k].n_cols, fake_data.col(i));
    }
    std::vector<arma::mat> true_folds(n_folds);
    for (size_t i = 0; i < true_data.n_cols; i++)
    {
        k = i % n_folds;
        true_folds[k].insert_cols(true_folds[k].n_cols, true_data.col(i));
    }
    
    /*Declaring necessary data structures. */
    arma::mat train_data;
    arma::mat train_data_f;
    arma::mat test_false_data;
    arma::mat test_true_data;
    arma::mat test_data;
    arma::mat test_data_f;
    arma::colvec y_pred;
    arma::colvec y_true;
    arma::mat a_p_r_f;
    
    for (size_t i = 0; i < n_folds; i++)
    {
        /*We select i-fold for testing. The others are used for training. */
        test_false_data = fake_folds[i];
        train_data.clear();
        for (size_t k = 0; k < n_folds; k++)
            if (k != i)
                train_data.insert_cols(train_data.n_cols, fake_folds[k]);

        train_data_f = MathUtil::get_instance()->drop_last_row(train_data);
        scaler.Fit(train_data_f);
        arma::mat train_data_transformed;
        scaler.Transform(train_data_f, train_data_transformed);
        
        clf->fit(train_data_transformed);
        
        for (size_t j = 0; j < n_folds; j++)
        {
            test_true_data = true_folds[j]; // Selecting one fold for testing.
            if (dataset_prefix == "datasets/WhatsApp/whats_br/Structured/")
                test_true_data = true_data;
            
            /*The two selected folders (one with true and other with fake data) are merged in one data structure*/
            test_data = arma::join_horiz(test_true_data, test_false_data);
            test_data_f = MathUtil::get_instance()->drop_last_row(test_data);
            arma::mat test_data_transformed;
            scaler.Transform(test_data_f, test_data_transformed);

            y_pred = clf->predict(test_data_transformed);
            y_true = MathUtil::get_instance()->get_last_row(test_data);
            a_p_r_f = classification_report(y_true, y_pred);
            accuracy_precision_recall_fscore = arma::join_vert(accuracy_precision_recall_fscore, a_p_r_f);
            if (dataset_prefix == "datasets/WhatsApp/whats_br/Structured/") break;
        }
    }
    arma::mat a_p_r_f_mean = arma::mean(accuracy_precision_recall_fscore, 0); // 0 = dim (mean of each column).
    arma::mat a_p_r_f_std = arma::stddev(accuracy_precision_recall_fscore, 0, 0); // 0s: Norm type, dim
    if (print_results)
    {
        std::cout << "Accuracy  = " << round(a_p_r_f_mean(0)) << "% \u00b1" << round(a_p_r_f_std(0)) << "%" << std::endl;
        std::cout << "Precision = " << round(a_p_r_f_mean(1)) << "% \u00b1" << round(a_p_r_f_std(1)) << "%" << std::endl;
        std::cout << "Recall    = " << round(a_p_r_f_mean(2)) << "% \u00b1" << round(a_p_r_f_std(2)) << "%" << std::endl;
        std::cout << "FScore    = " << round(a_p_r_f_mean(3)) << "% \u00b1" << round(a_p_r_f_std(3)) << "%" << std::endl;
    }
    delete clf;
    return a_p_r_f_mean(3);
}


void Model::manual_gridsearch_dcdistance(arma::colvec thresholds, 
                                         std::string dataset_prefix, 
                                         size_t n_folds,
                                         size_t feature_set, 
                                         std::vector<std::string> features)
{
    double fscore = 0.0;
    double temp;
    double t=0.1;
    switch (feature_set)
    {
        case FEATURES_CODE:
        case WORD2VEC_CODE:
            for (size_t i = 0; i < thresholds.n_rows; i++)
            {
                temp = classify(dataset_prefix, n_folds, new DCDistanceOCC(thresholds(i), EUCLIDEAN), false, feature_set, features);
                if (temp >= fscore)
                {
                    fscore = temp;
                    t = thresholds(i);
                }
            }
            classify(dataset_prefix, n_folds, new DCDistanceOCC(t, EUCLIDEAN), true, feature_set, features);
        break;
        case BOW_CODE:
        default:
            for (size_t i = 0; i < thresholds.n_rows; i++)
                {
                    temp = classify_bow(dataset_prefix, n_folds, new DCDistanceOCC(thresholds(i), COSINE), false);
                    if (temp >= fscore)
                    {
                        fscore = temp;
                        t = thresholds(i);
                    }
                }
                classify_bow(dataset_prefix, n_folds, new DCDistanceOCC(t, COSINE), true);
            break;
    }
    std::cout << "t: " << t << std::endl;
}

void Model::manual_gridsearch_occsvm(std::vector<std::string> nu, 
                                     std::vector<std::string> kernel, 
                                     std::vector<std::string> degree, 
                                     std::string dataset_prefix, 
                                     size_t n_folds,
                                     size_t feature_set, 
                                     std::vector<std::string> features)
{
    double fscore = 0.0;
    double temp;
    std::string nu_value;
    std::string kernel_value;
    std::string degree_value;
    switch (feature_set)
    {
        case FEATURES_CODE:
        case WORD2VEC_CODE:
            for (size_t k = 0; k < kernel.size(); k++)
            {
                for (size_t n = 0; n < nu.size(); n++)
                {
                    for (size_t d = 0; d < degree.size(); d++)
                    {
                        temp = classify(dataset_prefix, n_folds, new OCCSVM(kernel[k], degree[d], nu[n]), false, feature_set, features);
                        if (temp >= fscore)
                        {
                            fscore = temp;
                            nu_value = nu[n];
                            kernel_value = kernel[k];
                            degree_value = degree[d];
                        }
                        if (kernel[k].compare("1") != 0) // Degree is only used with poly kernel (which has code 1).
                            break;
                    }
                }
            }
            classify(dataset_prefix, n_folds, new OCCSVM(kernel_value, degree_value, nu_value), true, feature_set, features);
        break;
        case BOW_CODE:
        default:
            for (size_t k = 0; k < kernel.size(); k++)
            {
                for (size_t n = 0; n < nu.size(); n++)
                {
                    for (size_t d = 0; d < degree.size(); d++)
                    {
                        temp = classify_bow(dataset_prefix, n_folds, new OCCSVM(kernel[k], degree[d], nu[n]), false);
                        if (temp >= fscore)
                        {
                            fscore = temp;
                            nu_value = nu[n];
                            kernel_value = kernel[k];
                            degree_value = degree[d];
                        }
                        if (kernel[k].compare("1") != 0) // Degree is only used with poly kernel (which has code 1).
                            break;
                    }
                }               
            }
            classify_bow(dataset_prefix, n_folds, new OCCSVM(kernel_value, degree_value, nu_value), true);
            break;
    }    
    std::cout << "Kernel: " << kernel_value << std::endl;
    std::cout << "Degree: " << degree_value << std::endl;
    std::cout << "Nu: " << nu_value << std::endl;
}

void Model::manual_gridsearch_heif(std::vector<size_t> n_trees, 
                                   std::vector<size_t> sample_size,
                                   std::vector<size_t> extension_level,
                                   std::string dataset_prefix, 
                                   size_t n_folds,
                                   size_t feature_set,
                                   bool random_picks,
                                   std::vector<std::string> features)
{
    double fscore = 0.0;
    double temp;
    size_t n_trees_value = n_trees[0];
    size_t sample_size_value = sample_size[0];
    size_t extension_level_value = 0;
    switch (feature_set)
    {
        case FEATURES_CODE:
        case WORD2VEC_CODE:
            for (size_t t = 0; t < n_trees.size(); t++)
                {
                    for (size_t s = 0; s < sample_size.size(); s++)
                    {
                        for (size_t e = 0; e < extension_level.size(); e++)
                        {
                            temp = classify(dataset_prefix, n_folds, new HEIF(n_trees[t], sample_size[s], extension_level[e], random_picks), false, feature_set, features);
                            if (temp >= fscore)
                            {
                                fscore = temp;
                                n_trees_value = n_trees[t];
                                sample_size_value = sample_size[s];
                                extension_level_value = extension_level[e];
                            }
                        }     
                    }
                }
                classify(dataset_prefix, n_folds, new HEIF(n_trees_value, sample_size_value, extension_level_value, random_picks, 0, 0.3, 42, true), true, feature_set);
        break;
        case BOW_CODE:
        default:
            for (size_t t = 0; t < n_trees.size(); t++)
            {
                for (size_t s = 0; s < sample_size.size(); s++)
                {
                    for (size_t e = 0; e < extension_level.size(); e++)
                    {
                        temp = classify_bow(dataset_prefix, n_folds, new HEIF(n_trees[t], sample_size[s], extension_level[e], random_picks), false);
                        if (temp >= fscore)
                        {
                            fscore = temp;
                            n_trees_value = n_trees[t];
                            sample_size_value = sample_size[s];
                            extension_level_value = extension_level[e];
                        }
                    }
                }
            }
            classify_bow(dataset_prefix, n_folds, new HEIF(n_trees_value, sample_size_value, extension_level_value, random_picks, 0, 0.3, 42, true), true);
            break;
    }    
    std::cout << "n_trees: " << n_trees_value << std::endl;
    std::cout << "sample_size: " << sample_size_value << std::endl;
    std::cout << "extension level: " << extension_level_value << std::endl;
}
