//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>

#include "firmware/parameters.h"
#include "firmware/myproject.h"
#include "nnet_helpers.h"


int main(int argc, char **argv)
{

    //load input data from text file
    std::ifstream fin("tb_input_data.dat");

    std::string line;
    int e = 0;
    if (! fin.is_open())
    {
        std::cout << "Unable to open file" << std::endl;
        return 1;
    }

    std::ofstream outfile;
    outfile.open("tb_output_data.dat");
    while ( std::getline (fin,line) )
    {
        if( e%5000==0 ) std::cout << "Processing event " << e << std::endl;
        e++;
        char* cstr=const_cast<char*>(line.c_str());
        char* current;
        std::vector<float> arr;
        current=strtok(cstr," ");
        while(current!=NULL){
            arr.push_back(atof(current));
            current=strtok(NULL," ");
        }
        assert(arr.size() == N_INPUTS);

        input_t  data_str[N_INPUTS] = arr;
        result_t res_str[N_OUTPUTS] = {0};
        unsigned short size_in, size_out;
        myproject(data_str, res_str, size_in, size_out);

        for(int i=0; i<N_OUTPUTS; i++){
            outfile << res_str[i] << " ";
        }
        outfile << "\n";
    }
    fin.close();
    outfile.close();

    return 0;
}
