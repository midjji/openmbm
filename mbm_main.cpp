
#include <host_defines.h>
#include <mlib/utils/kitti.h>
#include <iostream>
#include <mlib/utils/files.h>
#include <mlib/cuda/mbm.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <mlib/utils/memmanager.h>
#include <mlib/cuda/devmemmanager.h>
#include <mlib/utils/cvl/MatrixNxM.hpp>
#include <mlib/utils/cvl/io.hpp>

#include <mlib/utils/congrats.h>




using std::cout;using std::endl;
using namespace cvl;








void testCongratsStereo(CongratsDataset cd);
void testKittiStereo(KittiDataset kd);




int main(int argc, const char* argv[]){











    if(argc<3){ // readme
        cout<<"Usage: "<<argv[0]<<" <dataset type> <path to dataset> ";
        cout<<"example: if the data is organized as follows: \n /store/datasets/kitti/dataset/sequences/00/image_0/\n";
        cout<<"argument 2 should be /store/datasets/kitti/dataset/"<<endl;
        cout<<"example:\n "<<argv[0]<<" kitti /store/datasets/kitti/dataset/ "<<endl;
        return 0;
    }
    // verify:
    if(!mlib::fileexists(argv[2])){cout <<"path: \""<<argv[1]<<"\" not found\n"; return 0;}


    if(std::string(argv[1])=="kitti")
        testKittiStereo(cvl::KittiDataset(argv[2]));
    else
        testCongratsStereo(CongratsDataset(argv[2]));
    cv::waitKey(0);
    return 0;
}



void testKittiStereo(KittiDataset kd){
    // main loop
    {
        cvl::MBMStereoStream mbm;

        int rows=kd.rows;
        int cols=kd.cols;
        // max size is fixed!
        mbm.init(64,rows,cols);

        std::vector<cv::Mat1b> imgs;

        for(int seq=0;seq<kd.sequences();++seq)
            for(int i=0;i<kd.images(seq);i+=1)
            {

                if(!kd.getImage(imgs,i,seq)) return;
                cout<<"read images"<<endl;
                cv::Mat1b left=imgs[0];
                cv::Mat1b right=imgs[1];

                if(left.rows!=rows ||left.cols!=cols)
                    cv::resize(left,left,cv::Size(cols,rows));
                if(right.rows!=rows ||right.cols!=cols)
                    cv::resize(right,right,cv::Size(cols,rows));

                cv::Mat1b disp=mbm(left,right);
                cout<<mbm.getTimer()<<endl;
                cv::imshow("disp0",disp);


                char key=cv::waitKey(10);
                if(key=='q') break;
            }
    }
}


void testCongratsStereo(CongratsDataset cd){



    // main loop
    {
        cvl::MBMStereoStream mbm;

        int rows=cd.rows;
        int cols=cd.cols;
        // max size is fixed!
        mbm.init(64,rows,cols);

        std::vector<cv::Mat3b> imgs;cv::Mat1f gt_disp;

        for(int seq=1;seq<cd.sequences();++seq)
            for(int i=0;i<cd.images(seq);i+=1)
            {

                if(!cd.getImage(imgs,gt_disp,i,seq)) return;

                cout<<"read images"<<endl;
                cv::Mat1b left,right;
                cv::cvtColor(imgs[0],left,cv::COLOR_BGR2GRAY);
                cv::cvtColor(imgs[1],right,cv::COLOR_BGR2GRAY);

                cv::Mat1b disp=mbm(left,right);

                cv::imshow("disp0",disp);
                cv::imshow("gt disp",normalize01(gt_disp));

                char key=cv::waitKey(0);
                if(key=='q') break;
            }
    }
}















