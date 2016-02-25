#include <assert.h>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <mlib/cuda/cuda_helpers.h>

using std::cout;
using std::endl;


/**
 * @brief devicePointer
 * @param p the pointer
 * @return if the pointer is __null or allocated on the device, since __null can be either or
 * &p[5] works aleast for the length of the allocated data
 */
bool devicePointer(void* p){

    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes,p);
    if(p==nullptr)
        return false;
    return (attributes.memoryType == cudaMemoryTypeDevice);
}
std::string getCudaErrorMsg(const cudaError_t& error){
    return cudaGetErrorString(error);
}

bool worked(const cudaError_t& error,std::string msg){
    if(error==cudaSuccess)
        return true;

        cout<<msg<<": "<<getCudaErrorMsg(error);
return false;

}

template<class T>
__global__ void printKernel(cvl::MatrixAdapter<T> m)
{
    if(blockIdx.x==0)
        if(threadIdx.x==0)
            for(int r=0;r<m.rows;++r){
                printf("row: %i - ",r);
                for(int c=0;c<m.cols;++c)
                    printf("%0.2f, ",m(r,c));
                printf("\n",r);
            }
}

template<class T>
void printdev(cvl::MatrixAdapter<T>  m){
    cudaDeviceSynchronize();
    dim3 grid(1,1,1);
    dim3 threads(1,1,1);
    printKernel<<<grid,threads>>>(m);
    cudaDeviceSynchronize();
}
void printDev(cvl::MatrixAdapter<int>  m){printdev(m);}
void printDev(cvl::MatrixAdapter<float>  m){printdev(m);}
void printDev(cvl::MatrixAdapter<double>  m){printdev(m);}
void printDev(cvl::MatrixAdapter<unsigned char>  m){printdev(m);}
