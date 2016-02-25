
#include <vector>
#include <set>
#include <assert.h>
#include <cmath>
#include <algorithm>

#pragma once
#ifndef VECTOR_HELPERS_H
#define VECTOR_HELPERS_H

namespace mlib {





template<typename T> inline
T mean(const std::vector<T>& v)
{
	assert(v.size() > 0 && "mean of zero element vector requested");
	T m = v.at(0);
	//int n=0;
	for (uint i = 1; i < v.size(); ++i) {
		//if(!isnan(v[i])){
		m += v[i];
		//  ++n;
	  //}
	}
	return m /= (unsigned long) v.size();
}


template<typename T> inline
T median(std::vector<T> v)
{
	assert((v.size() > 0) && "median of zero element vector requested");
	if (v.size() == 1)
		return v[0];
	sort(v.begin(), v.end());
	return v.at((unsigned int)std::round(v.size() / 2));
}

template<typename T> inline T min(const std::vector<T>& v)
{
	assert((v.size() != 0) && "min of empty vector undefined");
	T m = v.at(0);
	for (uint i = 1; i < v.size(); ++i) {
		if (m > v[i])
			m = v[i];
	}
	return m;
}

template<typename T> inline T max(const std::vector<T>& v)
{
	assert((v.size() != 0) && "max of empty vector undefined");
	T m = v.at(0);
	for (uint i = 1; i < v.size(); ++i) {
		if (m < v[i])
			m = v[i];
	}
	return m;
}

template<typename T> inline T sum(const std::vector<T>& v) 
{
	assert((v.size() != 0) && "sum of empty vector undefined");
	T m = v.at(0);
	for (uint i = 1; i < v.size(); ++i) {
		m += v[i];
	}
	return m;
}

template<typename T> inline void minmax(const std::vector<T>& v, T& min, T& max)
{
	assert((v.size() != 0) && "minmax of empty vector undefined");
	min = v.at(0);
	max = v.at(0);
	for (uint i = 1; i < v.size(); ++i) {
		if (max < v[i])
			max = v[i];
		if (min > v[i])
			min = v[i];
	}
}


template<class T> std::vector<T> unique_filter(const std::vector<T>& vs)
{
	std::set<T> us; std::vector<T> rs; rs.reserve(vs.size());
	for (const auto& v : vs)
		us.insert(v);
	for (const auto& u : us)
		rs.push_back(u);
	return rs;
}

template<typename T> inline void
reverse(std::vector<T>& v) {
	std::vector<T> r; r.reserve(v.size());
	for (int i = v.size() - 1; i > -1; --i) {
		r.push_back(v[i]);
	}
	v = r;
}


template<typename T> inline void
keep_filter(const std::vector<bool>& keep, std::vector<T>& v)
{
	assert(v.size() == keep.size());
	std::vector<T> vf; vf.reserve(v.size());
	for (uint i = 0; i < v.size(); ++i) {
		if (keep[i])
			vf.push_back(v[i]);
	}
	v = vf;
}


}// end namespace mlib
#endif // VECTOR_HELPERS_H
