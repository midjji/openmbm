#pragma once
/** cvl::Matrix<T,Rows,Cols>
  *
  * A matrix class for small matrices and vectors,
  * assumes real scalars
  */

#include <cassert>
#include <cmath>
#include <mlib/utils/cvl/MatrixAdapter.hpp>
//#include <iostream>
//using std::cout;using std::endl;
#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif


namespace cvl {

///  primarily for geometry operations.
template<class T, unsigned int Rows, unsigned int Cols>
class Matrix
{
protected:

public:
    T _data[Rows * Cols];


    //// Element access ////////////
    /// Access element (row, col)
    __host__ __device__
    T& operator()(unsigned int row, unsigned int col)
    {
        assert(row < Rows);
        assert(col < Cols);
        return _data[row * Cols + col];
    }

    /// Const access to element (row, col)
    __host__ __device__
    const T& operator()(unsigned int row, unsigned int col) const
    {
        assert(row < Rows);
        assert(col < Cols);
        return _data[row * Cols + col];
    }

    /// Access element (i). Useful for vectors and row-major iteration over matrices.
    __host__ __device__
    T& operator()(unsigned int i)
    {
        assert(i<Rows*Cols);
        return _data[i];
    }

    /// Const access to element (i). Useful for vectors and row-major iteration over matrices.
    __host__ __device__
    const T& operator()(unsigned int i) const
    {
        assert(i<Rows*Cols);
        return _data[i];
    }


    /// Access element (i). Useful for vectors and row-major iteration over matrices.
    __host__ __device__
    T& operator[](unsigned int i)
    {
        assert(i<Rows*Cols);
        return _data[i];
    }

    /// Const access to element (i). Useful for vectors and row-major iteration over matrices.
    __host__ __device__
    const T& operator[](unsigned int i) const
    {
        assert(i<Rows*Cols);
        return _data[i];
    }

    /// Get a pointer to the matrix or vector elements. The elements are stored in row-major order.
    T* data()    {        return _data;    }

    /// Get a const pointer to the matrix or vector elements. The elements are stored in row-major order.
    __host__ __device__
    const T* data() const   {
        return _data;
    }
    __host__ __device__
    T* getRowPointer(unsigned int row){
        assert(row<Rows);
        return &((*this)(row,0));
    }
    __host__ __device__
    Matrix<T,1,Cols> Row(unsigned int row) {
        assert(row<Rows);
        return Matrix<T,1,Cols>( getRowPointer(row));
    }
    __host__ __device__
    Matrix<T,Cols,1> RowAsColumnVector(unsigned int row) {
        assert(row<Rows);
        return Matrix<T,Cols,1>( getRowPointer(row));
    }
    __host__ __device__
    Matrix<T,Rows,1> Col(unsigned int column)    {
        assert(column<Cols);
        Matrix<T,Rows,1> col;
        for(int r=0;r<Rows;++r)
            col[r]=(*this)(r,column);
        return col;
    }
    __host__ __device__
    void setRow(const T* data,unsigned int row){
        T* rowptr=getRowPointer(row);
        for(int i=0;i<Cols;++i)rowptr[i]=data[i];
    }
    __host__ __device__
    void setRow(const Matrix<T,Rows,1>& rowvec, unsigned int row){
        setRow(rowvec.begin(),row);
    }
    __host__ __device__
    const T* begin() const{return &_data[0];}
    __host__ __device__
    const T* end() const{return &_data[Cols*Rows];}
    __host__ __device__
    unsigned int size(){return Cols*Rows;}
    __host__ __device__
    unsigned int cols(){return Cols;}
    __host__ __device__
    unsigned int rows(){return Rows;}

    /// Default constructor
    __host__ __device__
    Matrix(){
    //    static_assert(Cols*Rows>0,"empty matrix?");
    }
    /// Default destructor
    __host__ __device__ ~Matrix(){}



    /**
     * @brief Matrix
     * @param first_val
     * @param remaining_vals
     *
     * n-coefficient constructor, e.g Matrix2f m(1,2,3,4);
     * optimizer removes all surperflous alloc and assign
     */

    template<class... S>
    __host__ __device__
    Matrix(T first_val, S... remaining_vals)
    {
        static_assert(Cols*Rows>0,"empty matrix?");
            // its possibly but horribly bug prone to initialize with a single value!

            static_assert(sizeof...(S) == Cols * Rows - 1, "Incorrect number of elemets given");
            T b[] = { first_val, T(remaining_vals)... };

            for (unsigned int i = 0; i < Rows * Cols; i++) {
                _data[i] = b[i];
            }

    }

    /**
     * @brief Matrix
     * @param coeffs
     *
     * Does not verify size
     * its a bit dangerous to leave out or default userHasCheckedSize
     * since int will cast to both scalars and pointers !
     * Matrixxx(0) well might give you a segfault without has...  !
     *
     */
    __host__ __device__
    Matrix(const T* coeffs, bool userHasCheckedSize)
    {
        static_assert(Cols*Rows>0,"empty matrix?");



        if(coeffs) // not nullptr
            for (unsigned int i = 0; i < Rows * Cols; i++) _data[i] = coeffs[i];
        else
            for (unsigned int i = 0; i < Rows * Cols; i++) _data[i] = T(0);

    }

    template<class U>
    __host__ __device__
    Matrix(Matrix<U,Rows,Cols> M)
    {
        Matrix& a = *this;
        for(int i=0;i<Rows*Cols;++i)
            a(i)=T(M(i));
    }




    //// Elementwise arithmetic operations ////////////

    /// Add the elements of another matrix (elementwise addition)
    __host__ __device__
    Matrix& operator+=(const Matrix& b)
    {
        Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            a(i) += b(i);
        }
        return a;
    }

    /// Subtract the elements of another matrix (elementwise subtraction)
    __host__ __device__
    Matrix& operator-=(const Matrix& b)
    {
        Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            a(i) -= b(i);
        }
        return a;
    }

    /// Multiply by a scalar
    __host__ __device__
    Matrix& operator*=(const T& s)
    {
        Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            a(i) *= s;
        }
        return a;
    }

    /// Divide by a scalar
    __host__ __device__
    Matrix& operator/=(const T& s)
    {
        Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            a(i) /= s;
        }
        return a;
    }

    /// Elementwise matrix addtion
    __host__ __device__
    Matrix operator+(const Matrix& b) const
    {
        Matrix c;
        const Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            c(i) = a(i) + b(i);
        }
        return c;
    }

    /// Elementwise matrix subtraction
    __host__ __device__
    Matrix operator-(const Matrix& b) const
    {
        Matrix c;
        const Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            c(i) = a(i) - b(i);
        }
        return c;
    }

    /// Element negation
    __host__ __device__
    Matrix operator-() const
    {
        Matrix b;
        const Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            b(i) = -a(i);
        }
        return b;
    }

    /// Multiply by a scalar
    __host__ __device__
    Matrix operator*(const T& s) const
    {
        const Matrix& a = *this;
        Matrix b=a;
        b*=s;
        return b;
    }

    /// Divide by a scalar
    __host__ __device__
    Matrix operator/(const T& s) const
    {
        Matrix b;
        const Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            b(i) = a(i) / s;
        }
        return b;
    }

    //// Constant initializers /////////////
    __host__ __device__
    void setAll(const T& val){
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            _data[i] = val;
        }
    }
    /// Set all elements to zero.
    __host__ __device__
    Matrix& setZero()
    {
        setAll(0);
        return *this;
    }

    /// Return a matrix or vector with all elements set to zero.
    /// equivalent to Matrix<>(0)
    static Matrix Zero()
    {
        return Matrix(0);
    }
    /// Return a matrix or vector with all elements set to one.
    static Matrix Ones()
    {
        return Matrix(1);
    }

    /// Return a diagonal matrix.
    template<class... S>
    static Matrix diagonal(T first_val, S... remaining_vals)
    {
        static_assert(Cols*Rows>0,"empty matrix?");
        Matrix m;
        m.setAll(0);

        if (sizeof...(S) == 0) {
            for (unsigned int i = 0; (i < Cols && i < Rows); ++i)  m(i,i) = first_val;
            return m;
        }
        else{
            static_assert(sizeof...(S)==0||((sizeof...(S) +1== Cols ) && (sizeof...(S) +1<=Rows)) ||
                          ((sizeof...(S) +1== Rows ) && (sizeof...(S) +1<=Cols))
                          , "Incorrect number of elemets given");
            T b[] = { first_val, T(remaining_vals)... };
            for (unsigned int i = 0; i < Cols && i<Rows; ++i)
                m(i,i) = b[i];
        }
        return m;
    }

    /// Return an identity matrix.
    static Matrix Identity()    {       return diagonal(1);    }








    //// Various matrix operations ///////////////////////

    /// Return the matrix transpose
    __host__ __device__
    Matrix<T, Cols, Rows> transpose() const
    {
        Matrix<T, Cols, Rows> b;
        const Matrix& a = *this;
        // slower than needed...
        for (unsigned int row = 0; row < Rows; row++) {
            for (unsigned int col = 0; col < Cols; col++) {
                b(col, row) = a(row, col);
            }
        }
        return b;
    }
    __host__ __device__
    T trace() const{
        T tr=T(0);
        for (unsigned int i = 0; (i < Cols && i < Rows); ++i) tr+=(*this)(i,i);
        return tr;
    }

    /// Matrix determinant
    __host__ __device__
    T determinant() const
    {
        static_assert(Rows == Cols,"Must be square matrix");
        static_assert(Rows != 1,"?");
        static_assert(Rows < 4,"not implemented");
        const Matrix& a = *this;

        if (Rows == 2 ) {
            return a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);
        }
        if (Rows == 3 ) {

            // Minors
            T M00 = a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1);
            T M10 = a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2);
            T M20 = a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0);

            return a(0, 0) * M00 + a(0, 1) * M10 + a(0, 2) * M20;
        }
        {


            // Fall back to Eigen for larger matrices
            //typedef Eigen::Map<const Eigen::Matrix<T, Rows, Cols>, Eigen::RowMajor> CMap;
            //return CMap(_data).determinant();
        }
        // return T(0);// not needed since the fnction is never compiled where this happens
    }

    /// Matrix inverse
    __host__ __device__
    Matrix inverse() const
    {
        static_assert(Rows == Cols,"Must be square matrix");
        static_assert(Rows <5,"Why are you trying to invert a big matrix");

        const Matrix& a = *this;
        if(Rows==1){  Matrix b;b(0)=T(1)/a(0); return b;}
        if (Rows == 2) {
            Matrix b;
            T idet = T(1) / determinant();
            b(0, 0) = a(1, 1) * idet;
            b(0, 1) = -a(0, 1) * idet;
            b(1, 0) = -a(1, 0) * idet;
            b(1, 1) = a(0, 0) * idet;
            return b;
        }
        if (Rows == 3) {

            Matrix M; // Minors
            T idet; // Determinant

            M(0, 0) = a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1);
            M(0, 1) = a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2);
            M(0, 2) = a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1);

            M(1, 0) = a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2);
            M(1, 1) = a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0);
            M(1, 2) = a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2);

            M(2, 0) = a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0);
            M(2, 1) = a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1);
            M(2, 2) = a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);

            idet =T(1)/( a(0, 0) * M(0, 0) + a(0, 1) * M(1, 0) + a(0, 2) * M(2, 0));

            return (M * idet);

        }
        if(Rows==4){
            T mat[16];
            T dst[16];
            T tmp[16];

            // temp array for pairs
            T src[16];

            //Copy all of the elements into the linear array
            //int k=0;
            //for(int i=0; i<4; i++)
            //	for(int j=0; j<4; j++)
            //		mat[k++] = matrix[i][j];

            for(int i=0; i<16; i++)
                mat[i] = _data[i];




            // array of transpose source rix
            T det;

            /*determinant*/
            /*transposematrix*/
            for(int i=0;i<4;i++) //>
            {
                src[i]=mat[i*4];
                src[i+4]=mat[i*4+1];
                src[i+8]=mat[i*4+2];
                src[i+12]=mat[i*4+3];
            }

            // calculate pairs for first 8 elements (cofactors)
            tmp[0]=src[10]*src[15];
            tmp[1]=src[11]*src[14];
            tmp[2]=src[9]*src[15];
            tmp[3]=src[11]*src[13];
            tmp[4]=src[9]*src[14];
            tmp[5]=src[10]*src[13];
            tmp[6]=src[8]*src[15];
            tmp[7]=src[11]*src[12];
            tmp[8]=src[8]*src[14];
            tmp[9]=src[10]*src[12];
            tmp[10]=src[8]*src[13];
            tmp[11]=src[9]*src[12];

            // calculate first 8 elements (cofactors)
            dst[0]=tmp[0]*src[5]+tmp[3]*src[6]+tmp[4]*src[7];
            dst[0]-=tmp[1]*src[5]+tmp[2]*src[6]+tmp[5]*src[7];
            dst[1]=tmp[1]*src[4]+tmp[6]*src[6]+tmp[9]*src[7];
            dst[1]-=tmp[0]*src[4]+tmp[7]*src[6]+tmp[8]*src[7];
            dst[2]=tmp[2]*src[4]+tmp[7]*src[5]+tmp[10]*src[7];
            dst[2]-=tmp[3]*src[4]+tmp[6]*src[5]+tmp[11]*src[7];
            dst[3]=tmp[5]*src[4]+tmp[8]*src[5]+tmp[11]*src[6];
            dst[3]-=tmp[4]*src[4]+tmp[9]*src[5]+tmp[10]*src[6];
            dst[4]=tmp[1]*src[1]+tmp[2]*src[2]+tmp[5]*src[3];
            dst[4]-=tmp[0]*src[1]+tmp[3]*src[2]+tmp[4]*src[3];
            dst[5]=tmp[0]*src[0]+tmp[7]*src[2]+tmp[8]*src[3];
            dst[5]-=tmp[1]*src[0]+tmp[6]*src[2]+tmp[9]*src[3];
            dst[6]=tmp[3]*src[0]+tmp[6]*src[1]+tmp[11]*src[3];
            dst[6]-=tmp[2]*src[0]+tmp[7]*src[1]+tmp[10]*src[3];
            dst[7]=tmp[4]*src[0]+tmp[9]*src[1]+tmp[10]*src[2];
            dst[7]-=tmp[5]*src[0]+tmp[8]*src[1]+tmp[11]*src[2];

            // calculate pairs for second 8 elements(cofactors)
            tmp[0]=src[2]*src[7];
            tmp[1]=src[3]*src[6];
            tmp[2]=src[1]*src[7];
            tmp[3]=src[3]*src[5];
            tmp[4]=src[1]*src[6];
            tmp[5]=src[2]*src[5];
            tmp[6]=src[0]*src[7];
            tmp[7]=src[3]*src[4];
            tmp[8]=src[0]*src[6];
            tmp[9]=src[2]*src[4];
            tmp[10]=src[0]*src[5];
            tmp[11]=src[1]*src[4];

            // calculate second 8 elements (cofactors)
            dst[8]=tmp[0]*src[13]+tmp[3]*src[14]+tmp[4]*src[15];
            dst[8]-=tmp[1]*src[13]+tmp[2]*src[14]+tmp[5]*src[15];
            dst[9]=tmp[1]*src[12]+tmp[6]*src[14]+tmp[9]*src[15];
            dst[9]-=tmp[0]*src[12]+tmp[7]*src[14]+tmp[8]*src[15];
            dst[10]=tmp[2]*src[12]+tmp[7]*src[13]+tmp[10]*src[15];
            dst[10]-=tmp[3]*src[12]+tmp[6]*src[13]+tmp[11]*src[15];
            dst[11]=tmp[5]*src[12]+tmp[8]*src[13]+tmp[11]*src[14];
            dst[11]-=tmp[4]*src[12]+tmp[9]*src[13]+tmp[10]*src[14];
            dst[12]=tmp[2]*src[10]+tmp[5]*src[11]+tmp[1]*src[9];
            dst[12]-=tmp[4]*src[11]+tmp[0]*src[9]+tmp[3]*src[10];
            dst[13]=tmp[8]*src[11]+tmp[0]*src[8]+tmp[7]*src[10];
            dst[13]-=tmp[6]*src[10]+tmp[9]*src[11]+tmp[1]*src[8];
            dst[14]=tmp[6]*src[9]+tmp[11]*src[11]+tmp[3]*src[8];
            dst[14]-=tmp[10]*src[11]+tmp[2]*src[8]+tmp[7]*src[9];
            dst[15]=tmp[10]*src[10]+tmp[4]*src[8]+tmp[9]*src[9];
            dst[15]-=tmp[8]*src[9]+tmp[11]*src[10]+tmp[5]*src[8];

            // calculate determinant
            det=src[0]*dst[0]+src[1]*dst[1]+src[2]*dst[2]+src[3]*dst[3];

            // calculate matrix inverse
            det=T(1)/det;

            for(int j=0;j<16;j++) //>
                dst[j]*=det;

            //Copy everything into the output
            Matrix<T,Rows,Cols> ret(dst,true);
            return ret;
        }
        {


            // Fall back to Eigen for larger matrices.
            //typedef Eigen::Map<Eigen::Matrix<T, Rows, Cols>, Eigen::RowMajor> Map;
            //typedef Eigen::Map<const Eigen::Matrix<T, Rows, Cols>, Eigen::RowMajor> CMap;
            //Map(b._data) = CMap(_data).inverse();
        }

        return (*this);
    }

    /// Matrix multiplication

    template<unsigned int N>
    __host__ __device__
    Matrix<T, Rows, N> operator*(const Matrix<T, Cols, N>& b) const
    {
        Matrix<T, Rows, N> c;
        const Matrix& a = *this;
        for (unsigned int row = 0; row < Rows; row++) {
            for (unsigned int col = 0; col < N; col++) {

                T sum = T(0);
                for (unsigned int i = 0; i < Cols; i++) {
                    sum += a(row, i) * b(i, col);
                }
                c(row, col) = sum;
            }
        }
        return c;
    }

    /*
     * implicit homogeneous multiplication
     * may cause gcc to incorrectly emit array subscript is about array bounds
     * the specific cases that make sense are written explicitly below
     */
    __host__ __device__
    Matrix<T, Rows -1, 1> operator*(const Matrix<T, Cols -1, 1>& b) const
    {

        static_assert((Rows==3 && Cols==3)||(Rows==4 && Cols==4),"only makes sense for some sizes");

        return ((*this)*(b.homogeneous())).dehom();
        /*
         *  faster but needs a test!
        Matrix<T, Rows -1, 1> out;
        T z=0;
        const Matrix& a = *this;
        for (unsigned int row = 0; row < Rows -1; row++) {
            T sum = T(0);
            for (unsigned int col = 0; col < Cols -1; ++col) {
                sum += a(row, col) * b(col, 0);
            }
            sum += a(row, Cols-1);
            c(row, 0) = sum;
        }
        // compute the final value!
        for (unsigned int col = 0; col < Cols -1; ++col) {
            z+=a(Rows-1,col);
        }
        z+=a(Rows-1, Cols-1);
        out*=T(1.0)/z;

        return out;
        */
    }



    __host__ __device__
    Matrix<T, Rows, Cols> perElementMultiply(const Matrix<T, Rows, Cols>& b) const
    {
        Matrix<T, Rows, Cols> out;
        const Matrix& a = *this;
        for (unsigned int row = 0; row < Rows; row++)
            for (unsigned int col = 0; col < Cols; col++)
                out(row,col)=a(row,col)*b(row,col);
        return out;
    }



    /// Compute the inner product of this and another vector
    template<unsigned int Rows2, unsigned int Cols2>
    __host__ __device__
    T dot(const Matrix<T, Rows2, Cols2>& b) const
    {
        static_assert((Cols == 1 || Rows == 1),"The dot product is only defined for vectors.");
        static_assert( (Cols2 == 1 || Rows2 == 1),"The dot product is only defined for vectors.");
        static_assert(Rows * Cols == Rows2 * Cols2,
                      "The vectors in a dot product must have the same number of elements.");

        T sum = T(0);
        const Matrix& a = *this;
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            sum += a(i) * b(i);
        }
        return sum;
    }


    /// Compute the cross product of this and another vector
    __host__ __device__
    Matrix cross(const Matrix& b) const
    {
        static_assert((Rows == 3 && Cols == 1) || (Rows == 1 && Cols == 3),
                      "The cross product is only defined for vectors of length 3.");

        const Matrix& a = *this;
        Matrix<T,3,1> c(
                    a(1) * b(2) - a(2) * b(1),
                    a(2) * b(0) - a(0) * b(2),
                    a(0) * b(1) - a(1) * b(0)
                    );
        return c;
    }

    /// Return the 3-element vector as cross product matrix
    __host__ __device__
    Matrix<T, 3, 3> crossMatrix() const
    {
        static_assert((Rows == 3 && Cols == 1) || (Rows == 1 && Cols == 3),
                      "The cross product matrix is only defined for vectors of length 3.");

        const Matrix& a = *this;
        Matrix<T, 3, 3> b(
                    0, -a(2), a(1),
                    a(2), 0, -a(0),
                    -a(1), a(0), 0
                    );
        return b;
    }

    /// The sum of all elements
    __host__ __device__
    T sum() const
    {
        const Matrix& a = *this;
        T sum = T(0);
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            sum += a(i);
        }
        return sum;
    }

    /// The sum of the squared elements
    __host__ __device__
    T squaredNorm() const
    {
        const Matrix& a = *this;
        T sum = T(0);
        for (unsigned int i = 0; i < Rows * Cols; i++) {
            sum += a(i) * a(i); // does not do the right thing for complex values
        }
        return sum;
    }

    /// The L2 norm
    __host__ __device__
    T norm() const
    {
        return std::sqrt(squaredNorm());
    }
    /// The vector L2 norm, with a different name
    __host__ __device__
    T squaredLength() const
    {
        static_assert(Cols == 1 || Rows == 1,
                      "length() is only defined for vectors. Use norm() with matrices.");
        return squaredNorm();
    }

    /// The vector L2 norm, with a different name
    __host__ __device__
    T length() const
    {
        static_assert(Cols == 1 || Rows == 1,
                      "length() is only defined for vectors. Use norm() with matrices.");
        return norm();
    }

    /// the matrix divided by its own L2-norm
    __host__ __device__
    void normalize()
    {
        (*this) *= (T(1) / norm());
    }
    Matrix normalized() const
    {
        return (*this) * (T(1) / norm());
    }

    //Homogeneous coordiates, its always the last row
    __host__ __device__
    Matrix<T, Rows - 1, Cols> hnormalized()
    {
        Matrix< T, Rows - 1, Cols> b;
        const Matrix& a = *this;
        for (unsigned int col = 0; col < Cols; col++) {
            for (unsigned int row = 0; row < Rows - 1; row++) {
                b(row, col) = a(row, col) / a(Rows - 1, col);
            }
        }
        return b;
    }
    __host__ __device__
    Matrix<T, Rows - 1, Cols> dehom()
    {
        return hnormalized();
    }

    __host__ __device__
    Matrix< T, Rows + 1, Cols> homogeneous() const
    {
        Matrix< T, Rows + 1, Cols> b;
        const Matrix& a = *this;
        for (unsigned int col = 0; col < Cols; col++) {
            for (unsigned int row = 0; row < Rows; row++) {
                b(row, col) = a(row, col);
            }
            b(Rows, col) = T(1.0);
        }
        return b;
    }
    //inplace line normalization:
    __host__ __device__
    void  lineNormalize(){
        static_assert((Rows==3 && Cols == 1)||(Rows==1 && Cols == 3),"Line norm only defined for 3 vectors here");
        if((Rows==3 && Cols == 1)||(Rows==1 && Cols == 3)){
            if(_data[2]<0)
                (*this)*=T(1.0)/std::sqrt(_data[0]*_data[0] + _data[1]*_data[1]);
            else
                (*this)*=-T(1.0)/std::sqrt(_data[0]*_data[0] + _data[1]*_data[1]);
        }
    }
    /// Test whether the matrix or vector has at least one NaN element
    bool isnan() const
    {
        const Matrix& a = *this;
        for (int i = 0; i < Rows * Cols; ++i) {
            if(std::isnan(a(i))) return true;
        }
        return false;
    }
    bool isinf() const
    {
        const Matrix& a = *this;
        for (int i = 0; i < Rows * Cols; ++i) {
            if(std::isinf(a(i))) return true;
        }
        return false;
    }
    inline bool isnormal()const{// std::isnormal fails for 0!
        return !(isnan()||isinf());
    }


    __host__ __device__
    bool operator==(const Matrix& b) const
    {
        const Matrix& a = *this;

        for (unsigned int i = 0; i < Rows * Cols; i++) {
            if (a(i) != b(i)) return false;
        }
        return true;
    }
    // utility
    __host__ __device__
    T max(){
        const Matrix& A = *this;
        T maxv=A(0);
        for(int i=1;i<Rows*Cols;++i)
            if(maxv<A(i))maxv=A(i);
        return maxv;
    }
    __host__ __device__
    T min(){
        const Matrix& A = *this;
        T minv=A(0);
        for(int i=1;i<Rows*Cols;++i)
            if(minv>A(i))minv=A(i);
        return minv;
    }
    __host__ __device__
    void minmax(T& minv,T& maxv){
        const Matrix& A = *this;
        minv=maxv=A(0);
        for(int i=1;i<Rows*Cols;++i){
            if(maxv<A(i))maxv=A(i);
            if(minv>A(i))minv=A(i);
        }
    }
    /// Elementwise absolute value, only for reals
    __host__ __device__
    T absMax(){
        const Matrix& A = *this;
        T maxv=A(0);
        T tmp;
        for(int i=1;i<Rows*Cols;++i){
            tmp=A(i);
            if(tmp<0)       tmp=-tmp;
            if(maxv<tmp)    maxv=tmp;
        }
        return maxv;
    }



    // Numerics:
    bool isApprox(const Matrix& B, double prec) const
    {
        const Matrix& A = *this;

        double sum_a = abs(A).sum();
        double sum_b = abs(B).sum();
        double min_ab = (sum_a < sum_b ? sum_a : sum_b);

        return abs(B - A).sum() < (prec * prec) * min_ab; // same as Eigen::Matrix::isApprox()
    }







    // block matrix stuff:
    template<unsigned int rowstart,unsigned int colstart, unsigned int Height,unsigned int Width>
    __host__ __device__
    Matrix<T, Height, Width> getBlock() const
    {
        //static_assert(rowstart + Height < Rows,"Source matrix is too small");
        //static_assert(colstart + Width  < Cols,"Source matrix is too small");
        Matrix<T,Height, Width> out;
        for(unsigned int row=0;row<Height;++row)
            for(unsigned int col=0;col<Width;++col)
                out(row,col)=(*this)(row+rowstart,col+colstart);
        return out;
    }
    __host__ __device__
    Matrix<T,3,3> getRotationPart() const{
        return getBlock<0,0,3,3>();
    }
    __host__ __device__
    Matrix<T,3,1> getTranslationPart() const{
        return getBlock<0,3,3,1>();
    }






};




/// Free scalar-matrix multiplication s * matrix
template<typename T, unsigned int Rows, unsigned int Cols>
__host__ __device__
Matrix<T, Rows, Cols> operator*(const T& s, const Matrix<T, Rows, Cols>& a)
{
    Matrix<T, Rows, Cols> b;
    for (unsigned int i = 0; i < Rows * Cols; i++) {
        b(i) = a(i) * s;
    }
    return b;
}


/// Compute the inner product of this and another vector
template<class T, unsigned int Rows, unsigned int Cols,unsigned int Rows2, unsigned int Cols2>
__host__ __device__
T dot(const Matrix<T, Rows, Cols>& a, const Matrix<T, Rows2, Cols2>& b)
{
    static_assert((Cols == 1 || Rows == 1),"The dot product is only defined for vectors.");
    static_assert( (Cols2 == 1 || Rows2 == 1),"The dot product is only defined for vectors.");
    static_assert(Rows * Cols == Rows2 * Cols2,
                  "The vectors in a dot product must have the same number of elements.");

    T sum = T(0);
    for (unsigned int i = 0; i < Rows * Cols; i++) {
        sum += a(i) * b(i);
    }
    return sum;
}







/// Elementwise absolute value, only for reals
template<typename T, unsigned int Rows, unsigned int Cols>
__host__ __device__
Matrix<T, Rows, Cols> abs(const Matrix<T, Rows, Cols>& a)
{
    Matrix<T, Rows, Cols> b;
    for (unsigned int i = 0; i < Rows * Cols; i++) {
        if(a(i)<0)
            b(i) = -a(i);
        else
            b(i)=a(i);
    }
    return b;
}

/// Elementwise square root
template<typename T, unsigned int Rows, unsigned int Cols>
__host__ __device__
Matrix<T, Rows, Cols> sqrt(const Matrix<T, Rows, Cols>& a)
{
    Matrix<T, Rows, Cols> b;
    for (unsigned int i = 0; i < Rows * Cols; i++) {
        b(i) = sqrt(a(i));
    }
    return b;
}

template<class T, unsigned int Rows, unsigned int Cols>
Matrix<T,Cols,1> operator*(const VectorAdapter<T,Rows>& va,const Matrix<T,Rows,Cols>& m){
    Matrix<T,Cols,1> out(0);
    for(int row=0;row<Rows;++row)
        for(int col=0;col<Cols;++col){
            out(col)=va(col)*m(row,col);
        }
    return out;
}












template <typename T>
using Vector2 = Matrix<T, 2,1>;
template <typename T>
using Vector3 = Matrix<T, 3,1>;

template <typename T>
using Vector4 = Matrix<T, 4,1>;

template <typename T>
using Matrix2x2 = Matrix<T, 2,2>;
template <typename T>
using Matrix3x3 = Matrix<T, 3,3>;
template <typename T>
using Matrix3x4 = Matrix<T, 3,4>;
template <typename T>
using Matrix4x4 = Matrix<T, 4,4>;


///Size specific functions
template<class T>
__host__ __device__
Vector3<T> cross(const Vector3<T>& a, const Vector3<T>& b){
    return a.cross(b);
}
template<class T>
__host__ __device__
Vector3<T> getLineFrom2Points(const Vector2<T>& a,
                              const Vector2<T>& b){
    Vector3<T> line=a.homogeneous().cross(b.homogeneous());
    line.lineNormalize();
    return line;
}






template<class T>
__host__ __device__
Matrix4x4<T> get4x4(const Vector3<T>& t){
    return Matrix4x4<T>(0,0,0,t(0),
                        0,0,0,t(1),
                        0,0,0,t(2),
                        0,0,0,T(1));
}

template<class T>
__host__ __device__
Matrix4x4<T> get4x4(const Matrix3x3<T>& R){
    return Matrix4x4<T>(R(0,0),R(0,1),R(0,2),0,
                        R(1,0),R(1,1),R(1,2),0,
                        R(2,0),R(2,1),R(2,2),0,
                        0     ,     0,     0,1);
}
template<class T>
__host__ __device__
Matrix4x4<T> get4x4(const Matrix3x3<T>& R,const Vector3<T>& t){
    return Matrix4x4<T>(R(0,0),R(0,1),R(0,2),t(0),
                        R(1,0),R(1,1),R(1,2),t(1),
                        R(2,0),R(2,1),R(2,2),t(2),
                        0     ,     0,     0, 1   );
}
template<class T>
__host__ __device__
Matrix3x4<T> get3x4(const Matrix3x3<T>& R,const Vector3<T>& t){
    return Matrix3x4<T>(R(0,0),R(0,1),R(0,2),t(0),
                        R(1,0),R(1,1),R(1,2),t(1),
                        R(2,0),R(2,1),R(2,2),t(2));
}






using Vector2f = Matrix<float, 2, 1>;
using Vector3f = Matrix<float, 3, 1>;
using Vector4f = Matrix<float, 4, 1>;

using  Vector2d = Matrix<double, 2, 1>;
using  Vector3d = Matrix<double, 3, 1>;
using  Vector4d = Matrix<double, 4, 1>;

using Matrix2f = Matrix<float, 2, 2>;
using Matrix3f = Matrix<float, 3, 3>;
using Matrix34f = Matrix<float, 3, 4>;
using Matrix4f = Matrix<float, 4, 4>;

using  Matrix2d = Matrix<double, 2, 2>;
using  Matrix3d = Matrix<double, 3, 3>;
using  Matrix34d = Matrix<double, 3, 4>;
using  Matrix4d = Matrix<double, 4, 4>;

using Matrix3x3D= Matrix3d;
using Matrix4x4D= Matrix4d;
using Matrix3x4D= Matrix34d;

using Vector2D= Vector2d;
using Vector3D= Vector3d;
using Vector4D= Vector4d;

} // namespace cvl
