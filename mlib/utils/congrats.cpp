#include <mlib/utils/congrats.h>
#include "mlib/utils/files.h"
#include <mlib/utils/string_helpers.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


#include <mlib/utils/mlibtime.h>
#include <opencv2/highgui.hpp>

using std::cout;using std::endl;
using namespace mlib;







namespace cvl{
CongratsDataset::CongratsDataset(std::string basepath){
    this->basepath=basepath;
}
std::string CongratsDataset::getseqpath(int sequence){
    return basepath+mlib::toZstring(sequence,4)+"/";
}
bool CongratsDataset::getImage(std::vector<cv::Mat3b>& images,cv::Mat1f& disp, int number, int sequence){
    // cycle
cout<<sequence<<" "<<number<<endl;
    sequence=sequence % seqimgs.size();
    if(seqimgs.at(sequence)==0) return false;
    number=number % seqimgs.at(sequence);

    std::string path=getseqpath(sequence);
    images.clear();


    cv::Mat1b left,right;

    for(int cam=0;cam<2;++cam){
        std::string ipath=path+"image/" + mlib::toZstring(cam,2) + "/"+ mlib::toZstring(number+1,4)+".png";

        if(!mlib::fileexists(ipath,false)) {cout<<"image nr: "<<cam<<" not found: "<<ipath <<endl; return false;}
        cv::Mat3b image=cv::imread(ipath,cv::IMREAD_COLOR);

        //if(image.rows!=rows ||image.cols!=cols)            cv::resize(image,image,cv::Size(cols,rows));// resize?
        images.push_back(image);
    }

    std::string ipath=path+"gt/depth/00/"+mlib::toZstring(number+1,4)+".exr";
    if(!mlib::fileexists(ipath,false)) {cout<<"disparity nr: "<<number<<" not found: "<<ipath <<endl; return false;}
    disp=cv::imread(ipath,cv::IMREAD_ANYDEPTH);
    cout<<"disp: "<<disp.rows<<" "<<disp.cols<<endl;
    float minv,maxv;
    minmax(disp, minv,  maxv);
    cout<<"minmax " <<minv<<" "<<maxv<<endl;
    return true;
}
int CongratsDataset::sequences(){return seqimgs.size();}
int CongratsDataset::images(int sequence){return seqimgs.at(sequence);}



void testCongrats(std::string basepath){
    CongratsDataset kd(basepath);
    std::vector<cv::Mat3b> imgs; cv::Mat1f disp;

    for(int seq=1;seq<kd.sequences();++seq)
        for(int i=0;i<kd.images(seq);i+=1){

            if(!kd.getImage(imgs,disp,i,seq)) continue;

            cv::Mat1f normdisp=normalize01(disp);
            cv::imshow("Left",imgs[0]);
            cv::imshow("Right",imgs[1]);
            cv::imshow("Disp",normdisp);
            mlib::sleep(0.030);
            cv::waitKey(100);
        }
 mlib::sleep(1);
}





}// end namespace cvl
