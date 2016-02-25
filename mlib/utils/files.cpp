#include "mlib/utils/files.h"


#include <fstream>

#include <iostream>

#ifdef WITH_BOOST
    #include <boost/filesystem.hpp>
    #include <boost/filesystem/operations.hpp>
    #include <boost/filesystem/path.hpp>
#endif

using std::cout;using std::endl;


// the dependency on boost is troublesome. optionally disable it with reduced functionality.


namespace mlib{

void convertToNative(std::string& name){
#ifdef WITH_BOOST
    boost::filesystem::path p(name);
	//auto tmp = p.native();
	name = p.generic_string();
#endif
}
/* //4x faster filecheck on linux
bool file_exists(const std::string& name){
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}
*/
bool fileexists(std::string filename, bool verboseiffalse){

    convertToNative(filename);
    std::ifstream ifile(filename.c_str());
    bool exist=(ifile.is_open());
    if(!exist && verboseiffalse)
        std::cout<<"\nFile not found: "+filename<<endl;
    ifile.close();
    return exist;
}

}//end namespace mlib
