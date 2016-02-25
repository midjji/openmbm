#include <mlib/utils/kitti.h>
#include "mlib/utils/files.h"
#include <mlib/utils/string_helpers.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <mlib/utils/mlibtime.h>
#include <opencv2/imgproc.hpp>
using std::cout;using std::endl;
namespace cvl{
KittiDataset::KittiDataset(std::string basepath){
    this->basepath=basepath;
}
std::string KittiDataset::getseqpath(int sequence){
    return basepath+"sequences/"+mlib::toZstring(sequence,2)+"/";
}
bool KittiDataset::getImage(std::vector<cv::Mat1b>& images, int number, int sequence){
    // cycle
    sequence=sequence % seqimgs.size();
    number=number % seqimgs.at(sequence);

    std::string path=getseqpath(sequence);
    images.clear();
    cv::Mat1b left,right;
    std::string lpath=path+"image_0/"+mlib::toZstring(number,6)+".png";
    std::string rpath=path+"image_1/"+mlib::toZstring(number,6)+".png";
    if(!mlib::fileexists(lpath,false)) {cout<<"left image not found: "<<lpath <<endl; return false;}
    if(!mlib::fileexists(rpath,false)) {cout<<"right image not found: "<<rpath<<endl; return false;}

    left = cv::imread(lpath,cv::IMREAD_GRAYSCALE);
    right= cv::imread(rpath,cv::IMREAD_GRAYSCALE);
    // some rare images are of incorrect size, wonder why& if the calibration accounts for it
    if(left.rows!=rows ||left.cols!=cols)
        cv::resize(left,left,cv::Size(cols,rows));
    if(right.rows!=rows ||right.cols!=cols)
        cv::resize(right,right,cv::Size(cols,rows));
    images.push_back(left);
    images.push_back(right);
    return true;
}
int KittiDataset::sequences(){return seqimgs.size();}
int KittiDataset::images(int sequence){return seqimgs.at(sequence);}

void testKitti(std::string basepath){
    KittiDataset kd(basepath);
    std::vector<cv::Mat1b> imgs;

    for(int seq=0;seq<kd.sequences();++seq)
        for(int i=0;i<kd.images(seq);i+=1){
            if(!kd.getImage(imgs,i,seq)) continue;

            cv::imshow("Kitti Left",imgs[0]);
            cv::imshow("Kitti Right",imgs[1]);


            cv::waitKey(100);
        }
 mlib::sleep(1);
}
}// end namespace cvl
