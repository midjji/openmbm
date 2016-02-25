#pragma once
#include <mlib/utils/mlibtime.h>
#include <host_defines.h>
#include <mlib/utils/cvl/MatrixAdapter.hpp>

#include <mutex>
#include <memory>
#include <opencv2/core.hpp>
#include <mlib/cuda/devmemmanager.h>
namespace cvl{



typedef unsigned char uchar;



class DevMemManager;
class DevStreamPool;

/**
 * @brief The MBMStereoStream class
 * Stream class for the MBM stereo example,
 * initialize, then use, thread safe, blocking,allows most image sizes
 */
class MBMStereoStream{
public:

    void init(int disparities, int rows, int cols);
    cv::Mat1b operator()(cv::Mat1b Left,cv::Mat1b Right);
    mlib::Timer getTimer();
private:
    std::mutex mtx; // shared memory requires sync
    std::shared_ptr<DevMemManager> dmm=nullptr;
    std::shared_ptr<DevStreamPool> pool=nullptr;
    int disparities,cols,rows;
    cvl::MatrixAdapter<float> costs;
    cvl::MatrixAdapter<uchar> disps,disps2;
    cvl::MatrixAdapter<uchar> L0,R0;

    std::vector<cvl::MatrixAdapter<int>> adiffs;
    std::vector<cvl::MatrixAdapter<int>> sats;
    cvl::MatrixAdapter<cvl::MatrixAdapter<int>> satsv;



    bool inited=false;
    bool inner_median_filter=false;
    bool showdebug=false;
    mlib::Timer timer,mediantimer;
    mlib::Timer cumSumRowtimer,cumSumColtimer;

};

// cv::Mat owns the pointer
template<class T> MatrixAdapter<T> convertFMat(cv::Mat_<T> M){
    return MatrixAdapter<T>((T*)M.data,M.rows,M.cols,M.step/sizeof(T));
}


template<class T>
/**
  * @brief convert2Mat
  * @param M
  * @return
  *
  * cv::Mat does not take ownership!
  * oh for fucks sake opencv lacks take ownership of pointer!
  */
cv::Mat_<T> convert2Mat(MatrixAdapter<T> M){
    return cv::Mat_<T>(M.rows,M.cols,M.data,M.stride*sizeof(T));
}


// takes ownership
template<class T>
cv::Mat_<T> download2Mat(std::shared_ptr<DevMemManager>& dmm,
                         MatrixAdapter<T>& disps){
    cv::Mat_<T> big(disps.rows,disps.stride);
    MatrixAdapter<T> tmp((T*)big.data,disps.rows,disps.cols,disps.stride);
    dmm->download(disps,tmp);
    return big(cv::Rect(0,0,disps.cols,disps.rows));
}






template<class T> void print(cv::Mat_<T> img){
    for(int r = 0;r < img.rows;++r){
        std::cout<<"row: "<<r<<" - ";
        for(int c = 0;c < img.cols;++c)
            std::cout<<img(r,c)<<", ";
        std::cout<<"\n";
    }
}
template<class T> void print(cvl::MatrixAdapter<T> img){
    for(int r = 0;r < img.rows;++r){
        std::cout<<"row: "<<r<<" - ";
        for(int c = 0;c < img.cols;++c)
            std::cout<<img(r,c)<<", ";
        std::cout<<"\n";
    }
}





}// end namespace cvl
