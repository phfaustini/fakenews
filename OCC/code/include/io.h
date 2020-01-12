#ifndef IO_H
#define IO_H

#include <string>
#include <fstream>
#include <iostream>
#include <armadillo>
#include "csv.h"

class IO {
public:
   static IO* get_instance();
   bool file_exists(std::string& filepath);
   bool write_file(std::string& filepath, std::string& str);
   void append_file(std::string& filepath, std::string& str);
   arma::colvec read_file_as_colvec(std::string& filepath);
   arma::mat load_generic_dataset(std::string& filename);
   std::vector<std::string> load_text_dataset(std::string& filename);
private:
   IO();
   static IO* pSingleton;		// singleton instance
};

#endif