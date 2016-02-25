#pragma once
#ifndef Time_H
#define Time_H

#include <chrono>
#include <vector>
#include <iostream>


namespace mlib{
typedef std::chrono::nanoseconds NS_dur;
#ifdef __GNUG__ //gcc and c++
typedef std::chrono::time_point<std::chrono::steady_clock,NS_dur > NS_tp;
#else
typedef std::chrono::time_point<std::chrono::steady_clock, NS_dur > NS_tp;
//typedef std::chrono::time_point<std::chrono::system_clock, NS_dur > NS_tp;
#endif
enum TIME_TYPE{TIME_S,TIME_MS,TIME_US,TIME_NS};


std::string getIsoTime();
std::string getIsoDate();
std::string getIsoDateTime();

/**
 * @brief The Time class
 * A simplified convenient way to manage the std::chrono time system
 */
class Time{
public:
    long int ns=0;
    std::string toStr() const;
    std::string toStr(TIME_TYPE type) const;
    Time& operator+=(const Time& rhs);
    Time& operator/=(const Time& rhs);
    Time& operator/=(long unsigned int n);
    Time(long double nr,TIME_TYPE type);
    Time(long int ns);    
    double getSeconds(){double sec = ns/1000000000.0; return sec;}
    double getMilliSeconds(){double sec = ns/1000000.0; return sec;}
    void setSeconds(double sec){this->ns = (long)(sec*1000000000.0);}
    Time();
    ~Time();
};
bool operator==(const Time& lhs, const Time& rhs);
bool operator!=(const Time& lhs, const Time& rhs);
bool operator< (const Time& lhs, const Time& rhs);
bool operator> (const Time& lhs, const Time& rhs);
bool operator<=(const Time& lhs, const Time& rhs);
bool operator>=(const Time& lhs, const Time& rhs);
Time operator+ (const Time& lhs,const Time& rhs);
Time operator- (const Time& lhs,const Time& rhs);


std::ostream& operator<<(std::ostream &os, const Time& t);


void sleep(double seconds);
void sleep_ms(double milliseconds);
void sleep_us(double microseconds);



/**
 * @brief The Timer class
 * High precision Timer
 *
 * Timing is accurate in us to ms range, may be accurate in ns dep on implementation but not likely 
 *
 */
class Timer{

    NS_tp mark;
    NS_tp recall;
    NS_dur dur;
    std::vector<Time> ts;// Timestamps

public:

    Timer();
    void tic();
    /**
     * @brief toc
     * @return Time in nanoseconds
     */
    Time toc();
    std::string toStr() const;
    void show(std::string title=std::string("Timer = "), bool full=true);
    void write(std::string path, std::string name);
    // clears earlier measurements
    void clear();
    std::vector<Time> getTimes();       // alias for Time


};
std::ostream& operator<<(std::ostream &os, const Timer& t);

}// end namespace mlib



#endif // Time_H
