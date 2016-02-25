#include <mlib/cuda/devstreampool.h>
#include <mlib/cuda/cuda_helpers.h>
namespace cvl{
DevStreamPool::DevStreamPool(int size){ // 32 is the maximum number of async anyways...

    streams.resize(size);
    for(int i=0;i<size;++i)
        worked(cudaStreamCreate(&streams[i])); // can crash but, if so with exit(1) error..., takes some time... dont allow copy? yeah
}
DevStreamPool::~DevStreamPool(){
    synchronize(); // might stall but more informative error if so than obscure device segfault...
    for(int i=0;i<streams.size();++i) cudaStreamCreate( &streams[i]);
}
void DevStreamPool::synchronize(){
    for(int i=0;i<streams.size();++i)
        cudaStreamSynchronize(streams[i]);
}
void DevStreamPool::synchronize(uint i){
    cudaStreamSynchronize(streams[i % streams.size()]);
}
}
