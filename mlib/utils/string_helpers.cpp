#include "mlib/utils/string_helpers.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <mlib/utils/vector.h>

using namespace std;

namespace mlib
{
// string helpers

double str2double(char *str)
{
    std::istringstream ss(str);
    double d;
    ss >> d;
    return d;
}

int str2int(char *str)
{
    std::istringstream ss(str);
    int d;
    ss >> d;
    return d;
}

std::string toLowerCase(std::string in)
{
    std::string data = in;
    std::transform(data.begin(), data.end(), data.begin(), ::tolower);
    return data;
}

std::string toStr(char val)
{
    std::ostringstream ss("");
    ss << val;
    return ss.str();
}

std::string toStr(uint val)
{
    std::ostringstream ss("");
    ss << val;
    return ss.str();
}

std::string toStr(int val)
{
    std::ostringstream ss("");
    ss << val;
    return ss.str();
}
	
std::string toStr(long int val)
{
    std::ostringstream ss("");
    ss << val;
    return ss.str();
}

std::string toStr(float val, int res)
{
    long double v = val;
    return toStr(v, res);
}
	
std::string toStr(double val, int res)
{
    long double v = val;
    return toStr(v, res);
}
	
std::string toStr(long double val, int res)
{
    if (res > 10) res = 10;

    std::ostringstream ss("");
    if (res == 0) {
        ss << val;
    } else {
        ss << std::setprecision(res) << val;
    }
    return ss.str();
}

std::string toStr(bool val)
{
    std::ostringstream ss("");
    if (val)
        ss << true;
    else
        ss << false;
    return ss.str();
}
	
std::string toZstring(int i, int z)
{
    std::string Z = "";
    for (int a = 1; a < z; a++) {
        if (i < std::pow(10, a)) Z.append("0");
    }
    std::ostringstream ss("");
    ss << i;
    return Z.append(ss.str());
}

bool equal(std::string a, std::string b)
{
	return (a.compare(b) == 0);
}
	
bool hasNan(std::vector<double> xs)
{
    for (double x : xs)
        if (std::isnan(x)) return true;
    return false;
}

double median(std::vector<double> xs)
{
    if (xs.size() == 0) return 0;  // quiet nan
    sort(xs.begin(), xs.end());
    return xs.at(xs.size() / 2);
}

/**
 * @brief display
 * @param xs
 * @return a string showing common usefull statistics for the vector of doubles
 */
std::string display(std::vector<double> xs,bool showfull)
{
    if (xs.size() == 0) return "[]";
    std::stringstream ss;
    double avg = mean(xs);
    double medi = median(xs);
    double mi = min(xs);
    double ma = max(xs);
    ss << "mean median min max\n";
    ss << avg << " " << medi << " " << mi << " " << ma;

    if (xs.size() < 5 || showfull) {
        ss << "\n";
        for(int i=0;i<xs.size();++i){
            ss<<round(xs[i]*10)/10.0<<" ";
                if(i % 20 ==19)
                ss<<"\n";
        }
    }
    return ss.str();
}

double roundTo(double d, int digits)
{
    std::stringstream ss;
    ss << std::setprecision(digits) << d;
    double tmp;
    ss >> tmp;
    return tmp;
}
	
std::string s_printf(const char *fmt, ...)
{
	string s;

	va_list args;

	va_start(args, fmt);
	int n = vsnprintf(0, 0, fmt, args) + 1;
	va_end(args);

	s.resize(n, 'x');

	va_start(args, fmt);
	vsnprintf(const_cast<char*>(s.data()), n, fmt, args);
	va_end(args);

	return s;
}

}  // end namespace mlib
