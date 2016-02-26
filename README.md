# openmbm
a basic cuda accelerated implementation of the mbm stereo method 
(12ms per stereo pair on my machine ~5x faster than if build from plain opencv cuda functionalities)
The kernels are developed for compute capability 5+ and cuda 7.5 
I expect they will compile for lower versions but this needs to be explicitly enabled in the cmake file by changing
the CUDA_ARC_XXX defines. 

compile instructions
mkdir build
cd build 
cmake ..
make -j8

./mbm ...
