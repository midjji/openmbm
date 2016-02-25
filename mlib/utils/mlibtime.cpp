#include "mlibtime.h"
#include "mlib/utils/string_helpers.h"
#include <mlib/utils/vector.h>
#include <iomanip>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <thread>
#include <cmath>


using std::cout; using std::endl;
namespace mlib{

void sleep(double seconds){std::this_thread::sleep_for(std::chrono::milliseconds((int)(1000*seconds)));}
void sleep_ms(double milliseconds){std::this_thread::sleep_for(std::chrono::milliseconds((int)milliseconds));}
void sleep_us(double microseconds){std::this_thread::sleep_for(std::chrono::microseconds((int)microseconds));}

std::string getIsoDate(){
std::string datetime=getIsoDateTime();
return datetime.substr(0,10);
}
std::string getIsoTime(){
    std::string datetime=getIsoDateTime();
    return datetime.substr(11,8);
}

std::string getIsoDateTime()
{

    std::stringstream now;

    auto tp = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( tp.time_since_epoch() );
    size_t modulo = ms.count() % 1000;

    time_t seconds = std::chrono::duration_cast<std::chrono::seconds>( ms ).count();

#ifdef _MSC_VER

    struct tm T;
    localtime_s(&T, &seconds);
    now << std::put_time(&T, "%Y-%m-%d %H-%M-%S.");

#else

#define HAS_STD_PUT_TIME 0
#if HAS_STD_PUT_TIME
    now << std::put_time( localtime( &seconds ), "%Y-%m-%d %H-%M-%S." );
#else

    char buffer[25]; // holds "2013-12-01 21:31:42"

    // note: localtime() is not threadsafe, lock with a mutex if necessary
    if( strftime( buffer, 25, "%Y-%m-%d %H-%M-%S.", localtime( &seconds ) ) ) {
        now << buffer;
    }

#endif // HAS_STD_PUT_TIME

#endif //_MSC_VER

    // ms
    now.fill( '0' );
    now.width( 3 );
    now << modulo;

    return now.str();
}




namespace time{
static std::chrono::steady_clock clock;
}// end time

Time::Time(){
    ns=0;
}
Time::~Time(){}
Time::Time(long int ns){
    this->ns=ns;
}

Time::Time(long double d,TIME_TYPE type){

    switch(type){
    case TIME_S:
        ns=(long)(d*1000000000);
        break;
    case TIME_MS:
		ns = (long)(d * 1000000);
        break;
    case TIME_US:
		ns = (long)(d * 1000);
        break;
    case TIME_NS:
		ns = (long)d;
        break;
    }
}

Time& Time::operator+=(const Time& rhs){
    ns+=rhs.ns;
    return *this;
}
Time& Time::operator/=(const Time& rhs){
    ns/=rhs.ns;
    return *this;
}
Time& Time::operator/=(long unsigned int n){
    long double a=ns;
    long double b=n;
    a/=b;
    ns=(long)a;
    return *this;
}
bool operator==(const Time& lhs, const Time& rhs){ return lhs.ns==rhs.ns; }
bool operator!=(const Time& lhs, const Time& rhs){return !operator==(lhs,rhs);}
bool operator< (const Time& lhs, const Time& rhs){ return lhs.ns<rhs.ns; }
bool operator> (const Time& lhs, const Time& rhs){return  operator< (rhs,lhs);}
bool operator<=(const Time& lhs, const Time& rhs){return !operator> (lhs,rhs);}
bool operator>=(const Time& lhs, const Time& rhs){return !operator< (lhs,rhs);}

Time operator+ (const Time& lhs,const Time& rhs){
    return Time(lhs.ns+rhs.ns);
}
Time operator- (const Time& lhs,const Time& rhs){
    return Time(lhs.ns-rhs.ns);
}


std::string Time::toStr(TIME_TYPE type) const{
    long double d =ns;
    switch(type){
    case TIME_S:
        return mlib::toStr(std::round(d/1000000000L)).append("s");
    case TIME_MS:
        return mlib::toStr(std::round(d/1000000L)).append("ms");
    case TIME_US:
        return mlib::toStr(std::round(d/1000L)).append("us");
    case TIME_NS:
        return mlib::toStr(std::round(d)).append("ns");
    default:
#undef NDEBUG
        assert(false && "unsupported value");
        return "error";
#define NDEBUG
    }
}
std::string Time::toStr() const{

    std::stringstream ss;
    if(ns<20000){
        ss<<ns<<"ns";
        return ss.str();
    }
    if(ns<20000000){
        ss<<ns/1000<<"us";
        return ss.str();
    }
    if(ns<20000000000){
        ss<<ns/1000000<<"ms";
        return ss.str();
    }
        ss<<ns/1000000000<<"s";return ss.str();



    /*
    long double d= ns;

    d/=1000000000;
    if(d < 0.000001L){

    }

    if(d < 0.001L)
        return mlib::toStr((long int)std::round(d*1000000L)).append("us");

    if(d < 1L)
        return mlib::toStr((long int)std::round(d*1000L)).append("ms");

    if(d<3600)
        return mlib::toStr((long int)std::round(d)).append("s");

    return mlib::toStr(d/3600L,3).append("hours");
    */
}


std::ostream& operator<<(std::ostream& os,const Time& t){
    os<<t.toStr();
    return os;
}




Timer::Timer(){
    ts.reserve(100);
}
void Timer::tic(){
    mark=time::clock.now();
}
Time Timer::toc(){
    recall=time::clock.now();
    NS_dur diff = std::chrono::duration_cast<NS_dur>(recall - mark);

    Time d((long int)diff.count());
    ts.push_back(d);
    return d;
}
std::string Timer::toStr() const{
    if(ts.size()==0)
        return "No Beats";
    std::stringstream ss;
    if(ts.size()==1){
        ss<<ts.at(0).toStr();
        return ss.str();
    }


    TIME_TYPE type=TIME_NS;
    Time m=min<Time>(ts);
    if(m>1000L)
        type =TIME_US;
    if(m>1000000L)
        type =TIME_MS;
    if(m>1000000000L)
        type =TIME_S;
    //ss<<std::setprecision(8)<<std::fixed();

    ss<<"total: "<<sum(ts)<<" average: "<<mean(ts)<<" median: "<<median(ts)<<" samples: "<<ts.size()<<" ";
    if(ts.size()<20){
        ss<<"\n[";
        for(uint i=0;i<ts.size();++i){
            ss<< ts[i].toStr(type)<<"; ";        // works in matlab
            //if(i!=(ts.size()-1))
            //ss<<"\n";
            if((i%50)==49)
                ss<<"\n";
        }
        ss<<"]; ";
        if(type==TIME_NS)
            ss<<"ns";
        if(type==TIME_US)
            ss<<"us";
        if(type==TIME_MS)
            ss<<"ms";
        if(type==TIME_S)
            ss<<"s";
    }

    return ss.str();
}
void Timer::show(std::string title,bool full){

    std::cout<<"Timer: "<<title<<"\n";
    if(ts.size()==0){
        std::cout<<title<<"[]"<<std::endl;
        return;
    }
    std::cout<<"Total: "<<sum(ts)<<std::endl;
    if(ts.size()>1){
        std::cout<<"Size   : " <<ts.size()        <<"\n";
        std::cout<<"Min    : " <<min<Time>(ts)    <<"\n";
        std::cout<<"Mean   : " <<mean<Time>(ts)   <<"\n";
        std::cout<<"Median : " <<median<Time>(ts) <<"\n";
        std::cout<<"Max    : " <<max<Time>(ts)    <<"\n";
    }


    if(full){
        std::cout<< title  <<" = "<<toStr();
    }
    std::cout<<std::endl;
}
void Timer::clear(){ts.clear();}
void Timer::write(std::string path,std::string name){
    std::ofstream fos;
    fos.open(path+name);
    // wierd errors prevent me from setting the path dynamically...
    //cout<<"path"<<path<<endl;
    //assert(path==std::string("/home/mikael/bigdisk/exjobb/rtms/data/output/system/kitti-maps"));

    fos<<"close all;clear;clc;\n";
    fos<<"disp('Generating Times');\n";
    fos<<"cd '/home/mikael/bigdisk/exjobb/rtms/data/output/system/kitti-maps'\n";
    fos<<"disp('pwd')\n"<<"pwd\n";
    fos<<"addpath('/home/mikael/bigdisk/exjobb/rtms/data/output/system/kitti-maps/plot2svg_20120915/plot2svg_20120915')\n";
    fos<<"ms=0.001;\n";
    fos<<"t="<<toStr()<<";\n";
    name.pop_back();// remove ".m"
    name.pop_back();
    fos<<"writeplot(t,'../results/"+name+"');";
    fos<<"disp('Generating Times Done');\n";
    fos<<endl;
    fos.close();
}
std::vector<Time> Timer::getTimes(){
    return ts;
}

std::ostream& operator<<(std::ostream &os, const Timer& t){
    return os<<t.toStr();
}
}// end namespace mlib
