#include "../include/io.h"

IO* IO::pSingleton= NULL;

IO::IO() {}


IO* IO::get_instance()
{
	if (pSingleton == NULL)
		pSingleton = new IO();
	return pSingleton;
}

std::vector<std::string> IO::load_text_dataset(std::string& filename)
{
    std::vector<std::string> dataset;
    io::CSVReader<2, io::trim_chars<' '>, io::double_quote_escape<',','\"'> > in(filename); // https://github.com/ben-strasser/fast-cpp-csv-parser
    std::string text; double label;
    while(in.read_row(text, label))
    {
        dataset.push_back(text);
    }
    return dataset;
}

arma::mat IO::load_generic_dataset(std::string& filename)
{
    arma::mat dataset;
    dataset.load(filename);
    return dataset.t();
}

bool IO::file_exists(std::string& filepath)
{
    std::ifstream f(filepath.c_str());
    return f.good();
}

bool IO::write_file(std::string& filepath, std::string& str) // Overwrite existing file!
{
    std::ofstream f (filepath.c_str());
    if (f.is_open())
    {
        f << str;
        f.close();
        return true;
    }
    return false;
}

void IO::append_file(std::string& filepath, std::string& str)
{
    std::ofstream outfile;
    outfile.open(filepath, std::ios_base::app);
    outfile << str; 
}

arma::colvec IO::read_file_as_colvec(std::string& filepath)
{
    std::vector<int> v;

    std::string line;
    int value;
    std::ifstream myfile (filepath.c_str());
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            if (!line.empty())
            {
                value = std::stoi(line);
                v.push_back(value);
            }
        }
        myfile.close();
    }
    arma::colvec vec;
    vec.zeros(v.size());
    for (unsigned i = 0; i<v.size(); i++)
        vec(i) = v[i];
    return vec;
}