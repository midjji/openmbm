#include <mlib/cuda/devmemmanager.h>
namespace cvl{
DevMemManager::DevMemManager(){
    allocs.reserve(1024);
}
DevMemManager::~DevMemManager(){
    for(int i=0;i<allocs.size();++i){
        cudaFree(allocs[i]);allocs[i]=nullptr;
    }
}


void DevMemManager::synchronize(){
    pool.synchronize();
}
int DevMemManager::nextStream(){
    std::unique_lock<std::mutex> ul(mtx);
    return next++ % pool.streams.size();
}
}// end namespace cvl
