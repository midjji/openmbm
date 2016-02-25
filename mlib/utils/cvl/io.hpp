#pragma once



#include <mlib/utils/cvl/MatrixNxM.hpp>
#include <mlib/utils/cvl/MatrixAdapter.hpp>
#include <sstream>
#include <iostream>
namespace cvl{


// ideally read and write are identical, but they should also be informative:
// large matrixes should be summarized but the Matrix is for small ones
// this is for visualization purposes only, different in binary ones

template<class T, unsigned int Rows, unsigned int Cols>
std::ostream &operator <<(std::ostream &os, const cvl::Matrix<T,Rows,Cols>& v){

if(Rows>1 && Cols==1){
    return os<<v.transpose()<<"^T";
}
    os<<"[";
    for (unsigned int row = 0; row < Rows ; ++row){
        for (unsigned int col = 0; col < Cols; ++col){
            os<<v(row,col);
            if(col+1<Cols)
                os<<", ";
        }
        if(row+1<Rows)        os<<"\n ";
    }
    os << "]";
    return os;
}

template<class T, unsigned int Rows, unsigned int Cols>
std::istream &operator >>(std::istream &is, cvl::Matrix<T,Rows,Cols>& v){
    char toss;
    if(!(is>>toss)) return is;
    for(unsigned int row = 0; row < Rows ; ++row){
        for (unsigned int col = 0; col < Cols; ++col){
            if(!(is>>v(row,col))) return is;
            if(!(is>>toss)) return is;
        }

    }
    is>>toss;
    return is;
}






template<class T> std::string MatrixAdapter2String(MatrixAdapter<T> img){
    std::stringstream ss;

    for(int r = 0;r < img.rows;++r){
        ss<<"row: "<<r<<" - ";
        for(int c = 0;c < img.cols;++c)
            ss<<img(r,c)<<", ";
        ss<<"\n";
    }
    return ss.str();
}







// binary io // serialization:














} // end namespace cvl



