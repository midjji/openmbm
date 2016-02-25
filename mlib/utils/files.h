#pragma once
#include <string>
#include <vector>


namespace mlib{

// file helpers, mostly boost wrappers, some will work without boost but with reduced functionality
bool fileexists(std::string filename, bool verboseiffalse=true);


} // end namespace mlib

