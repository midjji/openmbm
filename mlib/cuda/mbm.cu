#include <mlib/cuda/mbm.h>

#include <iostream>
#include <mlib/cuda/cuda_helpers.h>
#include <opencv2/highgui.hpp>

#include <mlib/cuda/cuda_helpers.h>
#include <mlib/cuda/devmemmanager.h>
#include <mlib/utils/mlibtime.h>
#include <mlib/utils/cvl/MatrixNxM.hpp>

using std::cout;using std::endl;
using mlib::Timer;
namespace cvl{
template<class T>
cv::Mat1f normalize01(cv::Mat_<T> im){
    cv::Mat1f ret(im.rows,im.cols);
    T min,max;min=0;max=1;
    minmax(im, min, max);
    std::cout<<"cv minmax: "<<min<<" "<<max<<std::endl;
    for(int r=0;r<im.rows;++r)
        for(int c=0;c<im.cols;++c)
            ret(r,c)=((float)(im(r,c)-min))/((float)(max-min));
    return ret;
}




template<class T>
__global__ void setZeroK(MatrixAdapter<T> m){
    int row=blockIdx.x;
    int col=blockIdx.y*32+threadIdx.x;
    m(row,col)=0;
}

void setZeroDev(MatrixAdapter<uchar> m,cudaStream_t& stream){
    dim3 grid(m.rows,(m.cols+31)/32,1);
    dim3 threads(32,1,1);
    setZeroK<<<grid,threads,0,stream>>>(m);
}


template<class T> __host__ __device__
inline void split(T* data, T x, int& i, int& j){
    // assumes odd size vector
    // find a random value close to the median, mean is probably a good start
    while(i<=j){
        //scan from the left
        while(data[i]<x) ++i;
        //scan from the right
        while(x<data[j]) --j;

        //swap the two values
        if(i<=j){
            T tmp=data[i];
            data[i]=data[j];
            data[j]=tmp;
            i++;j--;
        }
    }
}

template<class T> __host__ __device__ inline T median(T* data, int size)
{
    int L = 0;
    int R = size-1;
    int k = size / 2;
    int i;int j;
    while (L < R)
    {
        T x = data[k];
        i = L; j = R;
        split(data, x,i,j);
        if (j < k)  L = i;
        if (k < i)  R = j;
    }
    return data[k];
}


template<unsigned int THREADS>
__global__ void medianfilter3x3(MatrixAdapter<uchar> in, MatrixAdapter<uchar> out)
{

    int row=blockIdx.x; // one per row -1
    int col=blockIdx.y*THREADS +threadIdx.x; // one per (col +31)/32
    if(col>in.cols) return;
    uchar loc[9];

    // just rely on the cache, yeah way faster than all attempts using shared mem...
    // data becomes effectively aligned this way anyways
    // and then the sorting starts earlier for some if the number of threads is large
    // the sorting is really slow...
    // I could read one row then sort those and so on? nope
    for(int r=0;r<3;++r)
        for(int c=0;c<3;++c)
            loc[r*3+c]=in(row+r,col+c);


    // bucket sort is too costly in registers
    // bubble isnt that bad in practice for this small matrixes and almost sorted data...
    /*
    for (int i = 0; i < 9; i++) {
        for (int j = i + 1; j < 9; j++) {
            if (loc[i] > loc[j]) { // swap?
                uchar tmp = loc[i];
                loc[i] = loc[j];
                loc[j] = tmp;
            }
        }
    }
    out(row+1,col+1)=loc[4];
    */

    // well it gets the same results but on larger matrixes its usually 4x quicksorts speed,
    // pretty much no improvement here though...
    // problem is this winds up beeing bound by the very worst of the arrays, so
    // I really want a good worst case variant...

    out(row+1,col+1)=median(loc,9);
}



#define CUMSUMCOLDIFFTHREADS 32
__global__ void cumSumColDiff(MatrixAdapter<uchar> L,
                              MatrixAdapter<uchar> R,
                              MatrixAdapter<int>  res){
    __shared__ int ocache[CUMSUMCOLDIFFTHREADS];
    int row=blockIdx.x;
    int cumsum=0;
    for(int col=0;col<res.cols;col+=CUMSUMCOLDIFFTHREADS){
        if(col+threadIdx.x<res.cols){
            ocache[threadIdx.x]=L(row,col+threadIdx.x);
            ocache[threadIdx.x]-=R(row,col+threadIdx.x);
            if(ocache[threadIdx.x]<0)ocache[threadIdx.x]=-ocache[threadIdx.x];
        }
        else
            ocache[threadIdx.x]=0;
        __syncthreads();

        if(threadIdx.x==0){ // simplest solution is so very often right ... // atleast now that the system is so ridicolously memory while running 60 at once
            for(int i=0;i<CUMSUMCOLDIFFTHREADS;++i){
                cumsum+=ocache[i];
                ocache[i]=cumsum;
            }
        }
        __syncthreads();
        if(col+threadIdx.x<res.cols)
            res(row,col+threadIdx.x)=ocache[threadIdx.x];
    }
}




#define ADIFFTHREADS 256
__global__ void adiff(MatrixAdapter<uchar> L,
                      MatrixAdapter<uchar> R,
                      MatrixAdapter<int>  res){


    int row=blockIdx.x;
    int col=blockIdx.y*ADIFFTHREADS +threadIdx.x;
    if(col>L.cols) return;
    int Lv=L(row,col);
    int Rv=R(row,col);
    int tmp=sqrtf((Lv-Rv)*(Lv-Rv));
    //if(tmp<0)tmp=-tmp;
    //if(tmp>255) tmp=255;
    res(row,col)=tmp;
}






#define CUMSUMCOLTHREADS 32
__global__

/**
         * @brief cumSumCol
         * @param adiff
         * @param out
         * requires one block per row and 32 threads...
         */
void cumSumCol(MatrixAdapter<int> adiff,
               MatrixAdapter<int>  out)
{



    // VERSION 1
    //read 32 using joint, then sum single,write 32 joint lower
    //mem and reg count... more than 2x as fasts as above in multiple sim kernel launches
    // beats using more shared mem..

    //__shared__ int acache[32]; // int here substantially improves performance because the conversions are made in parallel then!
    __shared__ int ocache[CUMSUMCOLTHREADS];
    int row=blockIdx.x;

    int cumsum=0;
    for(int col=0;col<adiff.cols;col+=CUMSUMCOLTHREADS){
        if(col+threadIdx.x<adiff.cols)
            ocache[threadIdx.x]=adiff(row,col+threadIdx.x);
        else
            ocache[threadIdx.x]=0;
        __syncthreads();


        if(threadIdx.x==0){
            // simplest solution is so very often right ...
            // atleast now that the system is so memory bound while running 64 at once
            for(int i=0;i<CUMSUMCOLTHREADS;++i){
                cumsum+=ocache[i];
                ocache[i]=cumsum;
            }
        }


        __syncthreads();
        if(col+threadIdx.x<adiff.cols)
            out(row,col+threadIdx.x)=ocache[threadIdx.x];
    }
}








#define CUMSUMROWTHREADS 64
__global__
void cumSumRow(MatrixAdapter<int>  csc)
{
    // VERSION 0
    // automatically becomes perfectly aligned
    int col=blockIdx.x*CUMSUMROWTHREADS+threadIdx.x;
    if(csc.cols<=col)        return;
    int prev=0;
    for(int row=0;row<csc.rows;++row){
        prev+=csc(row,col);
        csc(row,col)=prev;
    }
}


template<unsigned int THREADS> __global__
void cumSumRowV2(MatrixAdapter<MatrixAdapter<int>> meta)
{

    MatrixAdapter<int>  csc=meta(blockIdx.y,0);
    __syncthreads();
    // VERSION 1
    // automatically becomes perfectly aligned
    int col=blockIdx.x*THREADS+threadIdx.x;

    if(csc.cols<=col)        return;
    int prev=0;
    for(int row=0;row<csc.rows;++row){
        prev+=csc(row,col);
        csc(row,col)=prev;
    }
}





__device__
inline uint getBlockErrorSum(MatrixAdapter<int>& sat, int row, int col, int halfwidthrow, int halfwidthcol){
    // check for bad input

    //
    // -1 due to the offset in the unpadded sat
    int startr=row-halfwidthrow -1;
    int endrow=row+halfwidthrow ;
    int startc=col-halfwidthcol -1;
    int endcol=col+halfwidthcol ;


    if(sat.rows<endrow ||sat.cols<endcol||startc<0||startr<0){
        return 4294967295;
    }


    int A=sat(startr,startc);
    int B=sat(startr,endcol);
    int C=sat(endrow,startc);
    int D=sat(endrow,endcol);
    int err=D - B + A - C;
    return err;
}

__device__
inline uint getBlockErrorSum(MatrixAdapter<int>& sat, int row, int col, int halfwidth){
    return getBlockErrorSum(sat,row,col,halfwidth,halfwidth);
}


#define UPDATEDISPARITYTHREADS 32
__global__
/**
         * @brief updateDisparity
         * @param sat
         * @param disps
         * @param costs
         * one blockx per rows -32
         * one blocky per 32 cols
         * initializes costs and disps
         *
         */
void updateDisparity(MatrixAdapter<int> sat,
                     MatrixAdapter<uchar> disps,
                     MatrixAdapter<float> costs, int disp){

    int row=16+blockIdx.x; //one per row

    int col=blockIdx.y*UPDATEDISPARITYTHREADS + threadIdx.x;// 32 threads,blockIdx.y=(cols+31)/32
    uchar d=disp*4;
    //printf("%i",d);
    if(col>16 && col<sat.cols-16){

        float C7x7=getBlockErrorSum(sat,row,col,3,3);
        float C19x19=getBlockErrorSum(sat,row,col,9,9);

        float C61x1=1;
        if(row>31||row+31<sat.rows)
            C61x1=getBlockErrorSum(sat,row,col,30,1);
        float C1x61=1;
        if(col>31||col+31<sat.cols)
            C1x61=getBlockErrorSum(sat,row,col,1,30);
        float C61=C1x61;if(C61>C61x1)C61=C61x1;

        float err=C7x7*C19x19*C61;
        float C0=costs(row,col);
        if((C0>err)||disp==0){
            costs(row,col)=err;
            disps(row,col)=d;
        }
    }
}














mlib::Timer MBMStereoStream::getTimer(){return timer;}

void MBMStereoStream::init(int disparities, int rows, int cols){

    if(inited) return;
    cudaFree(0); //shoud ensure the cuda context is created...
    dmm=std::make_shared<DevMemManager>();
    pool=std::make_shared<DevStreamPool>(disparities);
    std::unique_lock<std::mutex> ul(mtx);// the images are allocated to new memory
    //cout<<"init "<<endl;


    this->disparities=disparities;
    this->rows=rows;
    this->cols=cols;
    costs=dmm->allocate<float>(rows,cols);
    disps=dmm->allocate<uchar>(rows,cols);
    disps2=dmm->allocate<uchar>(rows,cols);

    //printdev(disps);
    adiffs.reserve(disparities);
    sats.reserve(disparities);
    for(int i=0;i<disparities;++i)        adiffs.push_back(dmm->allocate<int>(rows,cols-i));
    for(int i=0;i<disparities;++i)        sats.push_back(dmm->allocate<int>(rows,cols-i));

    L0=dmm->allocate<uchar>(rows,cols);
    R0=dmm->allocate<uchar>(rows,cols);
    MemManager mm;
    MatrixAdapter<MatrixAdapter<int>> satsvhost=mm.allocate<MatrixAdapter<int>>(disparities,1);

    for(int i=0;i<disparities;++i) satsvhost(i,0)=sats[i];
    satsv=dmm->upload(satsvhost);



    // cv::Mat1b disp;
    dmm->synchronize();
    inited=true;
    // cout<<"init done"<<endl;
}

cv::Mat1f toMat1f(cv::Mat1i sat){
    cv::Mat1f ret(sat.rows,sat.cols);
    double min,max;cv::Point min_loc,max_loc;
    cv::minMaxLoc(sat, &min, &max, &min_loc, &max_loc);
    for(int r=0;r<sat.rows;++r)
        for(int c=0;c<sat.cols;++c)
            ret(r,c)=((float)(sat(r,c)-min))/((float)(max-min));
    return ret;
}

cv::Mat1b MBMStereoStream::operator()(cv::Mat1b Left,cv::Mat1b Right){
    std::unique_lock<std::mutex> ul(mtx);// the images are allocated to new memory
    if(!inited){        std::cerr<<"MBMStereoStream varian1 called before init!"<<endl;        exit(1);    }
    timer.tic();

    setZeroDev(disps,pool->streams[63]);
    dmm->upload(convertFMat(Left),L0);
    dmm->upload(convertFMat(Right),R0);
    dmm->synchronize();

    { // these kernels should be async => requries each in a stream of its own? YES!!!!!!
        Timer adifftimer;adifftimer.tic();
        for(int i=0;i<disparities;++i){

            int offset=i;
            // owned by the L0,R0 matrixes
            MatrixAdapter<uchar> L=L0.getSubMatrix(0,offset,L0.rows,L0.cols-offset);
            MatrixAdapter<uchar> R=R0.getSubMatrix(0,0,R0.rows,R0.cols-offset);
            dim3 grid(L.rows,(L.cols+ADIFFTHREADS-1)/ADIFFTHREADS,1);
            dim3 threads(ADIFFTHREADS,1,1);
            adiff<<<grid,threads,0,pool->streams[i]>>>(L,R,adiffs[i]);
        }
        pool->synchronize(); // wait untill its needed! or for testing enable

        adifftimer.toc();

        cout<<"adifftimer: "<<adifftimer<<endl;
    }
    cout<<"variant 1 1"<<endl;
    {
        // problem, this solution kills the cache!, well less of a issue after I fixed some very basic pipelining, and alignment issues
        // not sure what I do wrong but this takes longer than opencv bm method, despite my integral image beeing quite alot faster

        {
            cumSumColtimer.tic();
            for(int i=0;i<disparities;++i){

                //int blocks=(adiffs[i].rows+31)/32;
                int blocks=adiffs[i].rows;
                dim3 grid(blocks,1,1);
                dim3 threads(CUMSUMCOLTHREADS,1,1);
                cumSumCol<<<grid,threads,0,pool->streams[i]>>>(adiffs[i],sats[i]);

                if(showdebug){

                    pool->synchronize();
                    //printdev(sats[i]);

                    cv::Mat1f disp=toMat1f(download2Mat(dmm,sats[i]));
                    dmm->synchronize();

                    // print(tmp);
                    cv::imshow("colsum",disp);
                    cv::waitKey(0);
                }
            }
            pool->synchronize();// for speed testing


            cumSumColtimer.toc();
            cout<<"cumSumColtimer: "<<cumSumColtimer<<endl;
        }




        {
            cumSumRowtimer.tic();
            for(int i=0;i<disparities;++i){
                int blocks=(CUMSUMROWTHREADS-1+sats[i].cols)/CUMSUMROWTHREADS;
                dim3 grid(blocks,1,1);
                dim3 threads(CUMSUMROWTHREADS,1,1);
                // streams are sequential!
                cumSumRow<<<grid,threads,0,pool->streams[i]>>>(sats[i]);// computes the row sums
                if(showdebug){
                    pool->synchronize();
                    cv::Mat1f disp=toMat1f(download2Mat(dmm,sats[i]));
                    dmm->synchronize();

                    // print(tmp);
                    cv::imshow("rowsum",disp);
                    cv::waitKey(0);
                }
            }

            pool->synchronize();
            cumSumRowtimer.toc();
            cout<<"cumSumRowtimer: "<<cumSumRowtimer<<endl;
        }




    }
    //  cout<<"variant 1 3"<<endl;

    {
        Timer timer;timer.tic();
        for(int i=0;i<disparities;++i){
            int blockx=L0.rows-32;
            int blocky=(L0.cols+31)/32;
            dim3 grid(blockx,blocky,1);
            dim3 threads(32,1,1);
            pool->synchronize(i);
            updateDisparity<<<grid,threads,0,pool->streams[0]>>>(sats[i],disps,costs,i);
            if(inner_median_filter){

                mediantimer.tic();


                dim3 blocks(disps.rows-1,(disps.cols+63)/64,1);
                dim3 threads(64,1,1);
                medianfilter3x3<64><<<blocks,threads,0,pool->streams[0]>>>(disps,disps2);

                pool->synchronize(0);
                auto tmp=disps.data;                disps.data=disps2.data;                disps2.data=tmp;
                mediantimer.toc();
            }



            if(showdebug){
                pool->synchronize();
                cv::Mat1b disp=download2Mat(dmm,disps);
                dmm->synchronize();
                // --- Grid and block sizes
                cv::imshow("disparities",disp);
                cv::waitKey(0);
            }
        }
        pool->synchronize(0);
        timer.toc();
        cout<<"disparityTimer: "<<timer<<endl;
    }

    if(true){

        dim3 blocks(disps.rows-1,(disps.cols+63)/64,1);
        dim3 threads(64,1,1);
        medianfilter3x3<64><<<blocks,threads,0,pool->streams[0]>>>(disps,disps2);
        pool->synchronize(0);
        auto tmp=disps.data;                disps.data=disps2.data;                disps2.data=tmp;
    }

    // cant alloc with stride, but I can alloc a bigger one and return a submatrix...
    cv::Mat1b disp=download2Mat(dmm,disps);
    dmm->synchronize();
    timer.toc();
    cout<<"median timer: "<<mediantimer<<endl;
    return disp;
}












#if 0


/*

#include <cub/cub.cuh>

__global__ void ExampleKernel(...)
{
    // Specialize WarpScan for type int
    typedef cub::WarpScan<int> WarpScan;
    // Allocate WarpScan shared memory for one warp
    __shared__ typename WarpScan::TempStorage temp_storage;
    ...
    // Only the first warp performs a prefix sum
    if (threadIdx.x < 32)
    {
        // Obtain one input item per thread
        int thread_data = ...
        // Compute warp-wide prefix sums
                WarpScan(temp_storage).

        WarpScan(temp_storage).InclusiveSum(thread_data, thread_data);
    }
}

*/



__global__ void prescan(float *g_odata, float *g_idata, int n)
{
    extern __shared__ float temp[];  // allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;


    temp[2*thid] = g_idata[2*thid]; // load input into shared memory
    temp[2*thid+1] = g_idata[2*thid+1];

    for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;



        if (thid == 0) { temp[n - 1] = 0; } // clear the last element


        for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
        {
            offset >>= 1;
            __syncthreads();
            if (thid < d)
            {



                int ai = offset*(2*thid+1)-1;
                int bi = offset*(2*thid+2)-1;


                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
        __syncthreads();


        g_odata[2*thid] = temp[2*thid]; // write results to device memory
        g_odata[2*thid+1] = temp[2*thid+1];

    }
}

#endif






















}// end namespace cvl
