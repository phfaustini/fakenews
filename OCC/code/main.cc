#include "include/model.h"


void test(std::string dataset,
          std::string dataset_prefix, 
          size_t nfolds,
          arma::colvec dcdistanceocc_thresholds,
          std::vector<std::string> occsvm_kernel,
          std::vector<std::string> occsvm_degree,
          std::vector<std::string> occsvm_nu,
          std::vector<size_t> n_trees, 
          std::vector<size_t> sample_size, 
          std::vector<size_t> extension_level
          )
{
    Model m;
    std::cout << "##############" << std::endl;
    std::cout << "##############" << std::endl;
    std::cout << "DATASET " << dataset << std::endl;
    
    std::cout << std:: endl << "**************" << std:: endl;
    std::cout << "NBPC" << std:: endl;
    std::cout << "**************";
    std::cout << std::endl << "Custom features: " << std::endl;
    m.classify(dataset_prefix, nfolds, new NBPC(), true, FEATURES_CODE);
    std::cout << std::endl << "Word2Vec: " << std::endl;
    m.classify(dataset_prefix, nfolds, new NBPC(), true, WORD2VEC_CODE);
    std::cout << std::endl << "BOW: " << std::endl;
    m.classify_bow(dataset_prefix, nfolds, new NBPC(), true);

    std::cout << std:: endl << "**************" << std:: endl;
    std::cout << "DCDistanceOCC" << std:: endl;
    std::cout << "**************";
    std::cout << std::endl << "Custom features: " << std::endl;
    m.manual_gridsearch_dcdistance(dcdistanceocc_thresholds, dataset_prefix, nfolds, FEATURES_CODE);
    std::cout << std::endl << "Word2Vec: " << std::endl;
    m.manual_gridsearch_dcdistance(dcdistanceocc_thresholds, dataset_prefix, nfolds, WORD2VEC_CODE);
    std::cout << std::endl << "BOW: " << std::endl;
    m.manual_gridsearch_dcdistance(dcdistanceocc_thresholds, dataset_prefix, nfolds, BOW_CODE);

    std::cout << std:: endl << "**************" << std:: endl;
    std::cout << "ECOOCC" << std:: endl;
    std::cout << "**************";
    std::cout << std::endl << "Custom features: " << std::endl;
    std::string f = "results/best_k.txt";std::string s1;s1.append("F: features");s1.append("\n");IO::get_instance()->append_file(f, s1);
    m.classify(dataset_prefix, nfolds, new EcoOCC(300, EUCLIDEAN), true, FEATURES_CODE);
    std::cout << std::endl << "Word2Vec: " << std::endl;
    std::string s2;s2.append("F: Word2Vec");s2.append("\n");IO::get_instance()->append_file(f, s2);
    m.classify(dataset_prefix, nfolds, new EcoOCC(300, EUCLIDEAN), true, WORD2VEC_CODE);
    std::cout << std::endl << "BOW: " << std::endl;
    std::string s3;s3.append("F: BOW");s3.append("\n");IO::get_instance()->append_file(f, s3);
    m.classify_bow(dataset_prefix, nfolds, new EcoOCC(300, EUCLIDEAN), true);

    std::cout << std:: endl << "**************" << std:: endl;
    std::cout << "OneClassSVM" << std:: endl;
    std::cout << "**************";
    std::cout << std::endl << "Custom features: " << std::endl;
    m.manual_gridsearch_occsvm(occsvm_nu, occsvm_kernel, occsvm_degree, dataset_prefix, nfolds, FEATURES_CODE);
    std::cout << std::endl << "Word2Vec: " << std::endl;
    m.manual_gridsearch_occsvm(occsvm_nu, occsvm_kernel, occsvm_degree, dataset_prefix, nfolds, WORD2VEC_CODE);
    std::cout << std::endl << "BOW: " << std::endl;
    m.manual_gridsearch_occsvm(occsvm_nu, occsvm_kernel, occsvm_degree, dataset_prefix, nfolds, BOW_CODE);    

    std::cout << std:: endl << "**************" << std:: endl;
    std::cout << "EIF" << std:: endl;
    std::cout << "**************";
    std::cout << std::endl << "Custom features: " << std::endl;
    m.manual_gridsearch_heif(n_trees, sample_size, extension_level, dataset_prefix, nfolds, FEATURES_CODE, true);
    std::cout << std::endl << "Word2Vec: " << std::endl;
    m.manual_gridsearch_heif(n_trees, sample_size, extension_level, dataset_prefix, nfolds, WORD2VEC_CODE, true);
    std::cout << std::endl << "BOW: " << std::endl;
    m.manual_gridsearch_heif(n_trees, sample_size, extension_level, dataset_prefix, nfolds, BOW_CODE, true);

}


int main()
{
    /*Setting up hyperparameters.*/
    std::vector<double> temp;
    std::vector<std::string> occsvm_nu;
    for (size_t i=1; i<100; i+=5)
    {
        occsvm_nu.push_back(std::to_string( (double)i / 100));
        temp.push_back((double)i / 100);
    }
    arma::colvec dcdistanceocc_thresholds(temp);
    std::vector<std::string> occsvm_kernel = {"1", "2", "3"};
    std::vector<std::string> occsvm_degree = {"3"};
    std::vector<size_t> n_trees = {128, 256};
    std::vector<size_t> sample_size1 = {30}; std::vector<size_t> sample_size2 = {200}; std::vector<size_t> sample_size3 = {200}; std::vector<size_t> sample_size4 = {200}; std::vector<size_t> sample_size5 = {30}; std::vector<size_t> sample_size6 = {30}; 
    std::vector<size_t> extension_level = {0, 6, 13};
    size_t nfolds = 5;

    test("btv-lifestyle", "datasets/Websites/btv-lifestyle/Structured/", nfolds, dcdistanceocc_thresholds, occsvm_kernel, occsvm_degree, occsvm_nu, n_trees, sample_size1, extension_level);
    test("whats_br", "datasets/WhatsApp/whats_br/Structured/", 12, dcdistanceocc_thresholds, occsvm_kernel, occsvm_degree, occsvm_nu, n_trees, sample_size5, extension_level);
    test("fakenewsdata1", "datasets/Websites/fakenewsdata1_randomPolitics/Structured/", nfolds, dcdistanceocc_thresholds, occsvm_kernel, occsvm_degree, occsvm_nu, n_trees, sample_size6, extension_level);
    test("Bhattacharjee", "datasets/Websites/Bhattacharjee/Structured/", nfolds, dcdistanceocc_thresholds, occsvm_kernel, occsvm_degree, occsvm_nu, n_trees, sample_size2, extension_level);
    test("FakeBrCorpus", "datasets/Websites/FakeBrCorpus/Structured/", nfolds, dcdistanceocc_thresholds, occsvm_kernel, occsvm_degree, occsvm_nu, n_trees, sample_size3, extension_level);
    test("tweets_br", "datasets/Twitter/tweets_br/Structured/", nfolds, dcdistanceocc_thresholds, occsvm_kernel, occsvm_degree, occsvm_nu, n_trees, sample_size4, extension_level);
    
    return 0;
}
