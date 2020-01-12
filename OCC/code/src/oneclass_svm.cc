#include "../include/oneclass_svm.h"


OCCSVM::OCCSVM(std::string kernel, std::string degree, std::string nu) : GenericClassifier()
{
    this->kernel = kernel;
    this->nu = nu;
    this->degree = degree;
    this->svm_type = "2";
}


void OCCSVM::fit(arma::mat& X)
{
    std::string filetrain = "X_train_svm.train";
    bool converted = convert_arma_to_txt(X, filetrain);
    if (converted)
    {
        std::string command = "libsvm-3.23/svm-train -s " + this->svm_type + " -n " + this->nu + " -t " + this->kernel + " -d "+ this->degree+ " -q X_train_svm.train"; 
        int status = std::system(command.c_str());
        if (status == -1)
        {
            std::cout << "ERROR: Failed in call to system" << strerror(errno) << '\n';
            return;
        }
        else
            return;
    }
    std::cout << "ERROR: Failed to convert arma to txt" << strerror(errno) << '\n';
    return;
}


arma::colvec OCCSVM::predict(arma::mat& X)
{
    arma::colvec y_pred;
    y_pred.zeros(X.n_cols);
    std::string filetest = "X_train_svm.test";
    bool converted = convert_arma_to_txt(X, filetest);
    if(converted)
    {
        std::string command = "libsvm-3.23/svm-predict X_train_svm.test X_train_svm.train.model predicted.txt > /dev/null";
        int status = std::system(command.c_str());
        if (status == -1)
        {
            std::cout << "Error: " << strerror(errno) << '\n';
            return y_pred;
        }
        else
        {
            std::string predicted_file = "predicted.txt";
            arma::mat temp = IO::get_instance()->load_generic_dataset(predicted_file).t();
            y_pred = temp.col(0);
        }
    }
    return y_pred;
}


bool OCCSVM::convert_arma_to_txt(arma::mat& X, std::string& target_filename)
{
    std::string formatted_X;
    for (size_t i = 0; i < X.n_cols; i++)
    {
        formatted_X.append("1 "); // Libsvm ignore first column when testing. When training, all objects are inliers.
        for(size_t j = 1; j <= X.n_rows; j++)
        {
            formatted_X.append(std::to_string(j));
            formatted_X.append(":");
            formatted_X.append(std::to_string(X(j-1, i)));
            formatted_X.append(" ");
        }
        formatted_X.append("\n");
    }
    return IO::get_instance()->write_file(target_filename, formatted_X);
}