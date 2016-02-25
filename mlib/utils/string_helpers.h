#pragma once
#ifndef STRING_HELPERS_H
#define STRING_HELPERS_H
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>

typedef unsigned int uint;

namespace mlib
{
// string helpers

double str2double(char* str);
int str2int(char* str);

std::string toLowerCase(std::string in);

std::string toStr(char);
std::string toStr(int);
std::string toStr(uint);
std::string toStr(long int);
std::string toStr(bool);

std::string toStr(float, int res = 0);
std::string toStr(double, int res = 0);
std::string toStr(long double, int res = 0);

std::string toZstring(int i, int z = 5);

std::string display(std::vector<double> xs,bool showfull=false);

bool equal(std::string a, std::string b);

template <typename T>
std::string toStr(const std::vector<T>& v)
{
    std::stringstream ss;
    ss << "[";
    for (uint i = 0; i < v.size(); ++i) {
        ss << v[i];
        if (i < v.size() - 1) ss << ", ";
    }
    ss << "];\n";
    if (v.size() > 25) {
        ss << "v: (" << min(v) << ", " << mean(v) << ", " << median(v) << ", " << max(v) << "\n";
        return ss.str();
    }
    return ss.str();
}
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "Size: " << v.size() << "\n [";
    for (const T& t : v) os << t << ", ";
    os << "]";
    return os;
}

template <typename T1, typename T2>
inline std::ostream& operator<<(std::ostream& os, const std::map<T1, T2>& m)
{
    if (m.size() == 0) {
        os << "[]";
        return os;
    }
    os << "[";
    for (auto it = m.begin(); it != m.end(); ++it) {
        if (it != m.begin()) os << ", ";
        os << "(" << it->first << ": " << it->second << ")\n";
    }
    os << "]";
    return os;
}

/// sprintf with std::string output
std::string s_printf(const char *fmt, ...);

}  // end namespace mlib

#endif  // STRING_HELPERS_H
