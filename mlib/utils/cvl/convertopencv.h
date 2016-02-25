#pragma once
#include <mlib/utils/cvl/basic.hpp>
#include <opencv2/core.hpp>
namespace cvl{

template<class T, unsigned int Rows, unsigned int Cols>
cvl::Matrix<T,Rows,Cols> convert2Cvl(const cv::Mat_<T>& m){

    assert(m.rows==Rows);
    assert(m.cols==Cols);

    if(m.isContinuous())    return cvl::Matrix<T,Rows,Cols>((T*)m.data,true);

    cvl::Matrix<T,Rows,Cols> ret;// not safe for non continous matrixes
    for(int r=0;r<Rows;++r)
        for(int c=0;c<Cols;++c)
            ret(r,c)=m(r,c);
    return ret;
}

template<class T, unsigned int Rows, unsigned int Cols>
cv::Mat_<T> convert2Mat(const cvl::Matrix<T,Rows,Cols>& m){

    assert(m.rows==Rows);
    assert(m.cols==Cols);
    cv::Mat_<T> ret;
    for(int r=0;r<Rows;++r)
        for(int c=0;c<Cols;++c)
            ret(r,c)=m(r,c);
    return ret;
}

template<class T>
Vector2<T> convert2Cvl(const cv::Point_<T>& m){
    return Vector2<T>(m.x,m.y);
}

template<class T>
cv::Point_<T> convert2Point(const Vector2<T>& m){
    return cv::Point_<T>(m(0),m(1));
}
template<class T>
cv::Vec<T,2> convert2Vec(const Vector2<T>& m){
    return cv::Vec<T,2>(m(0),m(1));
}
template<class T>
cv::Vec<T,3> convert2Vec(const Vector3<T>& m){
    return cv::Vec<T,3>(m(0),m(1),m(2));
}




/// Will not collapse non continous matrixes, but will copy data
template<class T> cv::Mat_<T> toMat_(const cv::Mat& m){
    cv::Mat_<T> ret;ret(m.rows,m.cols,m.step/(sizeof(T)));
    if(m.isContinuous())
        std::memcpy(ret.data,m.data,m.rows*m.cols*sizeof(T));
    else
        for(int r=0;r<m.rows;++r)
            for(int c=0;c<m.cols;++c)
                ret(r,c)=m.at<T>(r,c);
    return ret;
}


cvl::Matrix<double,3,3> inline convert2Cvl3x3D(const cv::Mat& m){
    return convert2Cvl<double,3,3>(toMat_<double>(m));
}



}
// end namespace cvl
