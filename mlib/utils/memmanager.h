#pragma once
#include <mlib/utils/cvl/MatrixAdapter.hpp>
#include <vector>

namespace cvl{
/**
 * @brief The MemManager class
 * convenience class for dev memory management...
 * allocate takes care of stride alignment automatically, but returned matrixes arent guaranteed to be continuous
 */
class MemManager{
public:
    MemManager(){allocs.reserve(2048);}
    ~MemManager(){
        for(uint i=0;i<allocs.size();++i){
            delete allocs[i];allocs[i]=nullptr;
        }
    }

    template<class T>
    /**
     * @brief allocate
     * @param rows
     * @param cols
     * @return a automatically strided matrix! stride size is set to 256 which might be a bit much on cpu
     */
    MatrixAdapter<T> allocate(int rows, int cols){
        int stride=(256*((cols*sizeof(T)+255)/256))/sizeof(T);
        unsigned char* data=new unsigned char[stride*rows*sizeof(T)];
        allocs.push_back(data);
        return MatrixAdapter<T>((T*)data,rows,cols,stride);
    }
    template<class T>
    /**
     * @brief allocate
     * @param rows
     * @param cols
     * @param stride
     */
    MatrixAdapter<T> allocate(int rows, int cols, int stride){
        unsigned char* data=new unsigned char[stride*rows*sizeof(T)];
        allocs.push_back(data);
        return MatrixAdapter<T>((T*)data,rows,cols,stride);
    }
    template<class T> void manage(MatrixAdapter<T>& m){allocs.push_back((unsigned char*)m.data);}
    template<class T> void manage(T* data){allocs.push_back((unsigned char*)data);}


private:
    std::vector<unsigned char*> allocs;
};

}// end namespace cvl
