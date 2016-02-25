#pragma once
#include <cuda_runtime.h>
#include <vector>
namespace cvl{
/**
 * @brief The DevStreamPool class
 */
class DevStreamPool{
public:
    DevStreamPool(int size=32);
    ~DevStreamPool();
    void synchronize();
    void synchronize(uint i);
    std::vector<cudaStream_t> streams;
private:
};
}
