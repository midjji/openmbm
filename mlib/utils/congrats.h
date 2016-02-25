#pragma once

#include <mlib/utils/cvl/MatrixNxM.hpp>
#include <opencv2/core.hpp>
namespace cvl{


/**
 * @brief The CongratsDataset class
 * linux only convenient wrapper for the Congrats data,
 * sometimes the images are of different sizes? why?
 */
class CongratsDataset{
public:
    CongratsDataset(std::string basepath);
    std::string getseqpath(int sequence);
    bool getImage(std::vector<cv::Mat3b>& images,cv::Mat1f& disp, int number, int sequence);
    // not always true, but always returned by this thing
    int rows=384;
    int cols=1280;

    int sequences();
    int images(int sequence);


    std::string basepath;
    cvl::Matrix3x3<double> K=cvl::Matrix3x3<double>(718.856, 0, 607.1928,
                                                    0, 718.856, 185.2157,
                                                    0, 0, 1 );
    double baseline=(-3.861448/7.18856);
    std::vector<int> seqimgs={0,300};
};
/**
cv::Point min_loc, max_loc
bad idea though, it assumes alot of wierd shit
cv::minMaxLoc(your_mat, &min, &max, &min_loc, &max_loc);
**/
template<class T> bool minmax(cv::Mat_<T> im,T& minv, T& maxv){
    assert(im.rows>1);
    assert(im.cols>1);
    if(im.rows==0||im.cols==0)
        return true;
    minv=maxv=im(0,0);
    for(int r=0;r<im.rows;++r)
        for(int c=0;c<im.cols;++c){
            if(im(r,c)>maxv) maxv=im(r,c);
            if(im(r,c)<minv) minv=im(r,c);
        }
    return false;
}
template<class T>
cv::Mat1f normalize01(cv::Mat_<T> im){
    cv::Mat1f ret(im.rows,im.cols);
    T min,max;min=0;max=1;
    minmax(im, min, max);
    for(int r=0;r<im.rows;++r)
        for(int c=0;c<im.cols;++c)
            ret(r,c)=((float)(im(r,c)-min))/((float)(max-min));
    return ret;
}

void testCongrats(std::string basepath="/store/congrats/");
}// end namespace cvl
