#pragma once

#include <mlib/utils/cvl/MatrixNxM.hpp>
#include <opencv2/core.hpp>
namespace cvl{

/**
 * @brief The KittiDataset class
 * linux only convenient wrapper for the kitti data,
 * sometimes the images are of different sizes? why?
 */
class KittiDataset{
public:
    KittiDataset(std::string basepath);
    std::string getseqpath(int sequence);
    bool getImage(std::vector<cv::Mat1b>& images, int number, int sequence);
    // not always true, but always returned by this thing
    int rows=376;
    int cols=1241;

    int sequences();
    int images(int sequence);
private:
    std::string basepath;
    cvl::Matrix3x3<double> K=cvl::Matrix3x3<double>(718.856, 0, 607.1928,
                                                    0, 718.856, 185.2157,
                                                    0, 0, 1 );
    double baseline=(-3.861448/7.18856);
    std::vector<int> seqimgs={4541,1101,4661,801,271,2761,1101,1101,4071,1591,1201,921,1061,3281,631,1901,1731,491,1801,4981,831,2721};
};
void testKitti(std::string basepath="/store/datasets/kitti/dataset/");
}// end namespace cvl
