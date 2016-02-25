#pragma once

#include <assert.h>




#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

typedef unsigned int uint;
namespace cvl{

/**
 * @brief The MatrixAdapter class
 * A NxM non managed data Matrix
 */
template<class T>
class  MatrixAdapter{
public:

    __host__ __device__
    MatrixAdapter(){
        data=nullptr;
        rows=0;
        cols=0;
        stride=0;
    }
    __host__ __device__
    MatrixAdapter(T* data, uint  rows, uint  cols){
        this->data=data;
        this->cols=cols;
        this->rows=rows;
        this->stride=cols;
        assert(stride>=cols);

    }
    __host__ __device__
    MatrixAdapter(T* data,uint  rows, uint  cols, uint stride){
        this->data=data;
        this->cols=cols;
        this->rows=rows;
        this->stride=stride;
        assert(stride>=cols);
    }
    __host__ __device__
    MatrixAdapter(MatrixAdapter<T>& m, uint row, uint col, uint rows, uint cols){
        assert(row+rows<=this->rows);
        assert(col+cols<=this->cols);
        data=atref(row,col);
        this->cols=cols;
        this->rows=rows;
        stride=m.stride;
    }
    __host__ __device__ ~MatrixAdapter(){}

    __host__ __device__
    inline T& operator()( const int& row, const int& col){ return data[row*stride +col ]; }
    __host__ __device__
    inline const T& operator()( const int& row, const int& col ) const    { return data[row*stride +col ]; }

    __host__ __device__
    inline T& at(uint row, uint col){        assert(col<cols);        assert(row<rows);        return data[row*stride + col];    }
    // bilinear, slow implementation...

    __host__ __device__
    T* atref(uint row, uint col) const{        return &(data[row*stride+col]);    }

    // get the submatrix pointer...
    __host__ __device__
    MatrixAdapter<T> getSubMatrix(uint row, uint col) const{
        MatrixAdapter<T> m(atref(row,col),rows-row,cols-col,stride);
        return m;
    }
    __host__ __device__
    MatrixAdapter<T> getSubMatrix(uint row, uint col, uint rows, uint cols) const{
        MatrixAdapter<T> m(atref(row,col),rows,cols,stride);
        assert(row+rows<=this->rows);
        assert(col+cols<=this->cols);
        return m;
    }
    __host__ __device__
    MatrixAdapter<T> row(uint row) {
        assert(row<rows);
        MatrixAdapter<T> m(atref(row,0),1,cols,stride);
        return m;
    }
    __host__ __device__
    MatrixAdapter<T> col(uint col) {
        assert(col<cols);
        MatrixAdapter<T> m(atref(0,col),rows,1,stride);
        return m;
    }

    MatrixAdapter<T> clone(){
        T* data=new T[rows*cols];
        MatrixAdapter<T> m(data,rows,cols);
        for(uint row=0;row<rows;++row)
            for(uint col=0;col<cols;++col)
                m(row,col)=at(row,col);
        return m;
    }


    template<class T1> MatrixAdapter<T1> convert(){

        T1* data=new T1[rows*cols];
        MatrixAdapter<T1> m(data,cols,rows);

        for(uint x=0;x<cols;++x)
            for(uint y=0;y<rows;++y)
                m.set(x,y,(T1)at(x,y));
        return m;
    }


    bool isContinuous() const {return cols==stride;}

    uint rows;
    uint cols;
    uint stride;
    T* data=nullptr;
};


template<class T, unsigned int Size>
class VectorAdapter{
public:
    T* _data;
    VectorAdapter(){}
    VectorAdapter(T* data){_data=data;}
    ~VectorAdapter(){} // non managed
    /// Access element
    __host__ __device__
    T& operator()(unsigned int index)    {        assert( index < Size);            return _data[index];    }
    __host__ __device__
    T& operator[](unsigned int index)    {        assert( index < Size);            return _data[index];    }
    __host__ __device__
    const T& operator()(unsigned int index) const {        assert( index < Size);            return _data[index];    }
    __host__ __device__
    const T& operator[](unsigned int index) const {        assert( index < Size);            return _data[index];    }

    /// Get a pointer to the  vector elements.
    /// __host__ __device__
    T* data()    {        return _data;    }
    __host__ __device__
    const T* data() const {        return _data;    }
    unsigned int size(){return Size;}

};












}// end namespace cvl
