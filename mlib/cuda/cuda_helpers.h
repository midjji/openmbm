#pragma once
/**
  If any cuda kernel outputs nothing but otherwize seems to work,
 check if you are generating code for the appropriate compute cap
  */
/***
 * gtx 750 :
 * Cuda Compute capability 5.0
 * cores 640
 * Shared Memory / SM		64 KB
Register File Size / SM	256 KB	256 KB
Active Blocks / SM		32
Memory Clock		5400 MHz
Memory Bandwidth		86.4 GB/s
L2 Cache Size		2048 KB
*/

#include <host_defines.h>
#include <assert.h>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <mlib/utils/cvl/MatrixAdapter.hpp>





/**
 * @brief devicePointer
 * @param p the pointer
 * @return if the pointer is __null or allocated on the device, since __null can be either or
 * &p[5] works aleast for the length of the allocated data
 */
bool devicePointer(void* p);
std::string getCudaErrorMsg(const cudaError_t& error);
bool worked(const cudaError_t& error,std::string msg="");




template <class T>
T* cudaNew(int elements){
   T* data=nullptr;
    cudaError_t error;
    error = cudaMalloc((void **) &data, elements*sizeof(T));
    if (error != cudaSuccess)
    {
        std::cout<<"Failed to allocate memory on the device: size is: "<<sizeof(T)*elements/(1024*1024)<<"MB"<< "cuda error code is "<<(int)error<<" which means "<<getCudaErrorMsg(error)<<std::endl;
        exit(1);
    }
    return data;
}

/**
 * @brief copy
 * @param from
 * @param to
 * @param elems
 * copy from dev to host or vice versa automatically
 */
//void copy(float*  from, float*& to, unsigned int elements);
template<class T>
void copy(T*  from, T*& to, unsigned int elements){

    assert(from!=nullptr);// check sizes somehow?


    if(devicePointer(from)){
     //   std::cout<<"copy from dev: "<<elements<<std::endl;
        if(to==nullptr){

            to=new T[elements];
        }
        worked(cudaMemcpy(to,from, elements*sizeof(T), cudaMemcpyDeviceToHost));
    }else{
       // std::cout<<"copy to dev: "<<elements<<std::endl;
        if(to==nullptr)
            to=cudaNew<T>(elements);
        worked(cudaMemcpy(to, from, elements*sizeof(T), cudaMemcpyHostToDevice));
    }
    assert(to!=nullptr);
}


/**
 * @brief copy
 * @param from
 * @param to
 * @param elems
 * copy from dev to host or vice versa automatically
 * Async version
 */
template<class T>
void copy(T*  from, T*& to, unsigned int elements, cudaStream_t& stream){

    assert(from!=nullptr);// check sizes somehow?
    if(devicePointer(from)){
        //   std::cout<<"copy from dev: "<<elements<<std::endl;
        if(to==nullptr)            to=new T[elements];
        worked(cudaMemcpyAsync(to,from, elements*sizeof(T), cudaMemcpyDeviceToHost, stream));

    }else{
        // std::cout<<"copy to dev: "<<elements<<std::endl;
        if(to==nullptr)            to=cudaNew<T>(elements);
        worked(cudaMemcpyAsync(to,from, elements*sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    assert(to!=nullptr);
}




void printDev(cvl::MatrixAdapter<int>  m);
void printDev(cvl::MatrixAdapter<float>  m);
void printDev(cvl::MatrixAdapter<double>  m);
void printDev(cvl::MatrixAdapter<unsigned char>  m);














