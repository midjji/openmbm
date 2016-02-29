CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)
PROJECT(openmbm CXX C) # defines ${CMAKE_PROJECT_NAME}(top level) PROJECT_NAME(most recent)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/mlib/cmake;${CMAKE_MODULE_PATH}")
include(mlib/cmake/mlib-utils-common-scripts.cmake)
#build config
BuildConfig()
# Warning behaviour
WarningConfig()
# opt flags per compiler
OptimizationConfig()

#compiles mlib, mlib-cuda
getmlibfiles(${PROJECT_SOURCE_DIR})

cuda_add_executable(mbm mbm_main.cpp)
target_link_libraries(mbm ${LIBS})

