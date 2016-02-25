#pragma once
#include <host_defines.h>
#include <mlib/cuda/cuda_helpers.h>
#include <mlib/utils/cvl/MatrixAdapter.hpp>


#include <mutex>
#include <mlib/utils/memmanager.h>
#include <mlib/cuda/devstreampool.h>

namespace cvl{

/**
 * @brief The DevMemManager class
 * convenience class for dev memory management...
 * synchronize up and downloads, remember that alloc is synced!, ideally prealloc using allocate.
 * allocate takes care of stride alignment automatically, but returned matrixes wont nec be continuous
 */
class DevMemManager{
public:

    DevMemManager();
    ~DevMemManager();

    template<class T>
    /**
     * @brief allocate
     * @param rows
     * @param cols
     * @return a automatically strided matrix! Always use these on device!
     */
    MatrixAdapter<T> allocate(int rows, int cols){
        int stride=(256*((cols*sizeof(T)+255)/256))/sizeof(T);
        //stride=cols;
        T* data=cudaNew<T>(rows*stride);
        allocs.push_back((void*)data);
        //cout<<"stride: "<<stride<<" "<<cols<<endl;
        return MatrixAdapter<T>(data,rows,cols);
        //return MatrixAdapter<T>(data,rows,cols,stride);
    }


    template<class T>
    void upload(MatrixAdapter<T> hostMat, MatrixAdapter<T>& preAllocOut){
        // not const ref because sigh...

        assert(hostMat.rows==preAllocOut.rows);
        assert(hostMat.cols==preAllocOut.cols);

        if(hostMat.isContinuous()){// use this at all? yes! eliminates per thread overhead...
            copy<T>(hostMat.data, preAllocOut.data,preAllocOut.stride*preAllocOut.rows,pool.streams[nextStream()]);
        }
        else{

            for(int row=0;row<hostMat.rows;++row){
                T* rowptr=&preAllocOut(row,0);
                copy<T>(&hostMat(row,0), rowptr,preAllocOut.cols,pool.streams[nextStream()]); // cols instead of stride since its a submatrix we want !
            }
        }
    }

    template<class T>
    MatrixAdapter<T> upload(const MatrixAdapter<T>& M){
        MatrixAdapter<T> out=allocate<T>(M.rows,M.cols);
        upload(M,out);
        return out;
    }




    template<class T>
    // does not take ownership
    MatrixAdapter<T> download(MatrixAdapter<T> m){
        T* data=new T[m.rows*m.stride];
        MatrixAdapter<T> ret(data,m.rows,m.cols,m.stride);
        copy<T>(m.data,ret.data,m.rows*m.stride,pool.streams[0]);        
        return ret;
    }
    template<class T>
    // does not take ownership
    void download(MatrixAdapter<T> m,MatrixAdapter<T>& preAlloc){
        copy<T>(m.data,preAlloc.data,m.rows*m.stride,pool.streams[0]);
    }


    void synchronize();
    int nextStream();
private:
    DevStreamPool pool;
    std::mutex mtx;
    std::vector<void*> allocs;
    int next=0;
};
}// end namespace cvl
