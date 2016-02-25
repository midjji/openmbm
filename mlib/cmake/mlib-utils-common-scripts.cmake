
# common defines
set(line "<================================================================================>")
macro(PrintList list prepend)
    foreach(el ${list})
        message("${prepend} -- ${el}")
    endforeach()
endmacro()

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
        LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()















macro(SetVersion major minor patch)
    set(${CMAKE_PROJECT_NAME}_VERSION_MAJOR ${major})
    set(${CMAKE_PROJECT_NAME}_VERSION_MINOR ${minor})
    set(${CMAKE_PROJECT_NAME}_VERSION_PATCH ${patch})
    set(${CMAKE_PROJECT_NAME}_VERSION ${${CMAKE_PROJECT_NAME}_VERSION_MAJOR}.${${CMAKE_PROJECT_NAME}_VERSION_MINOR}.${${CMAKE_PROJECT_NAME}_VERSION_PATCH})

# Additionally, when CMake has found a XXXConfig.cmake, it can check for a
# XXXConfigVersion.cmake in the same directory when figuring out the version
# of the package when a version has been specified in the FIND_PACKAGE() call,
# e.g. FIND_PACKAGE(XXX [1.1.1] REQUIRED). The version argument is optional.
    if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/XXXConfigVersion.cmake")
        set(package_version ${${CMAKE_PROJECT_NAME}_VERSION})
        CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/cmake/XXXConfigVersion.cmake"
               "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}ConfigVersion.cmake" @ONLY)
    endif()
endmacro()


# Build configuration
macro(BuildConfig)
# Change the default build type from Debug to Release

# The CACHE STRING logic here and elsewhere is needed to force CMake
# to pay attention to the value of these variables.(for override)
    if(NOT CMAKE_BUILD_TYPE)
        MESSAGE("-- No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
        set(CMAKE_BUILD_TYPE Release CACHE STRING
            "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."  FORCE)
    else()
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            MESSAGE("\n${line}")
            MESSAGE("\n-- Build type: Debug. Performance will be terrible!")
            MESSAGE("-- Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
            MESSAGE("-- Add -DCMAKE_BUILD_TYPE=RelWithDebInfo to the CMake command line to get an faster build with symbols(-g).")
            MESSAGE("\n${line}")
        endif()
    endif()

# compiler specific:
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(languageused "-std=c++11")
        add_definitions(${languageused})
                else()
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP8")
    endif()



# WINDOWS EXTRA DEFINES
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
add_definitions(-D__STDC_LIMIT_MACROS)
endif()



# extra options
option(random_seed_zero "RANDOM_SEED_ZERO" ON)
if(random_seed_zero)
    add_definitions(-DRANDOM_SEED_ZERO)
endif()

endmacro()

macro(WarningConfig)
    #todo add compiler specific...
    option(WExtrawarnings "Extra warnings" ON)
    option(WError "Warnings are errors" OFF)
    if(CMAKE_COMPILER_IS_GNUCXX)  # GCC
        if(WError)
            set(warningoptions "${warningoptions}-Werror ")
        endif()


# GCC is not strict enough by default, so enable most of the warnings. but disable the annoying ones
        if(WExtrawarnings)

            set(warn "${warn} -Wall")
            set(warn "${warn} -Wextra")
            set(warn "${warn} -Wno-unknown-pragmas")
            set(warn "${warn} -Wno-sign-compare")
            set(warn "${warn} -Wno-unused-parameter")
            set(warn "${warn} -Wno-missing-field-initializers")
            set(warn "${warn} -Wno-unused")
            set(warn "${warn} -Wno-unused-function")
            set(warn "${warn} -Wno-unused-label")
            set(warn "${warn} -Wno-unused-parameter")
            set(warn "${warn} -Wno-unused-value")
            set(warn "${warn} -Wno-unused-variable")
            set(warn "${warn} -Wno-unused-but-set-parameter")
            set(warn "${warn} -Wno-unused-but-set-variable")
            set(warningoptions "${warningoptions}${warn}")
            # disable annoying ones ...
            list(APPEND warningoptions )
        endif()

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}${warningoptions}")
        # also no "and" or "or" ! for msvc
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-operator-names")
    endif()
endmacro()


macro(OptimizationConfig)
    if(CMAKE_COMPILER_IS_GNUCXX)  # GCC
        set(CMAKE_CXX_FLAGS_RELEASE "-march=native -mtune=native -O3 -Ofast -DNDEBUG")
    else()
        message("TODO: fix opt options on this compiler")
    endif()
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message("FOUND OPENMP")
    endif(OPENMP_FOUND)
endmacro()


# find opencv macro which fixes the damned opencv bad names and ensures its convenient to link to it even if it fails to find it automatically
macro(Add_Package_OpenCV)
    find_package( OpenCV 3 )
    if(OpenCV_FOUND)
        message("-- Found OpenCV version ${OpenCV_VERSION}")
        set(OpenCV_LIBRARIES ${OpenCV_LIBS})# fixes the name
        add_definitions(-DWITH_OPENCV)
# pulls in cuda options, mark them as advanced...

 mark_as_advanced(FORCE CUDA_BUILD_CUBIN)
 mark_as_advanced(FORCE CUDA_BUILD_EMULATION)
 mark_as_advanced(FORCE CUDA_HOST_COMPILER)
 mark_as_advanced(FORCE CUDA_SDK_ROOT_DIR)
 mark_as_advanced(FORCE CUDA_SEPARABLE_COMPILATION)
 mark_as_advanced(FORCE CUDA_TOOLKIT_ROOT_DIR)
 mark_as_advanced(FORCE CUDA_VERBOSE_BUILD)


    else()
        message("-- OpenCV Not Found, set the corresponding options")
        set(OpenCV_Include_A "" CACHE PATH "Path to opencv ie <path>/include/opencv")
        set(OpenCV_Include_B "" CACHE PATH "Path to opencv2 ie <path>/include/")
        set(OpenCV_INCLUDE_DIRS "${OpenCV_Include_A};${OpenCV_Include_B}")

        set(tofind "opencv_videostab;opencv_video;opencv_ts;opencv_superres;opencv_stitching;opencv_photo;opencv_ocl;opencv_objdetect;opencv_nonfree;opencv_ml;opencv_legacy;opencv_imgproc;opencv_highgui;opencv_gpu;opencv_flann;opencv_features2d;opencv_core;opencv_contrib;opencv_calib3d;imgcodec")
        set(OpenCV_LIBRARIES "")
        foreach(pth ${tofind})
            set(OpenCV_libpath_${pth} "OFF" CACHE FILEPATH "Filepath to ${pth}")

            if(EXISTS ${OpenCV_libpath_${pth}})
                list(APPEND OpenCV_LIBRARIES ${OpenCV_libpath_${pth}})
            endif()
        endforeach()
    endif()
    if(VERBOSE)
        message("-- Include directories: ${OpenCV_INCLUDE_DIRS}")
        message("-- OpenCV_Libraries:  ")
        PrintList("${OpenCV_LIBRARIES}" "    ")
    endif()
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    list(APPEND LIBS ${OpenCV_LIBRARIES})
endmacro()



macro(getmlibfiles basepath)


Add_Package_OpenCV()
Find_Package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11; -gencode arch=compute_50,code=sm_50;--ptxas-options=-v -O3)#--maxrregcount 32;





#file(GLOB_RECURSE files FOLLOW_SYMLINKS  "*.h" "*.hpp" "*.cpp" "*.cu")
#foreach(file ${files})
#message("list(APPEND mlibfiles \"${file}\" )")
#endforeach()

INCLUDE_DIRECTORIES(${basepath})



list(APPEND mlibfiles "${basepath}/mlib/utils/string_helpers.h" )

list(APPEND mlibfiles "${basepath}/mlib/utils/files.h" )




list(APPEND mlibfiles "${basepath}/mlib/utils/memmanager.h" )





list(APPEND mlibfiles "${basepath}/mlib/utils/mlibtime.h" )








list(APPEND mlibfiles "${basepath}/mlib/utils/cvl/io.hpp" )



list(APPEND mlibfiles "${basepath}/mlib/utils/cvl/MatrixNxM.hpp" )





list(APPEND mlibfiles "${basepath}/mlib/utils/cvl/convertopencv.h" )



list(APPEND mlibfiles "${basepath}/mlib/utils/cvl/MatrixAdapter.hpp" )

list(APPEND mlibfiles "${basepath}/mlib/utils/mlibtime.cpp" )
list(APPEND mlibfiles "${basepath}/mlib/utils/kitti.cpp" )
list(APPEND mlibfiles "${basepath}/mlib/utils/congrats.cpp" )

list(APPEND mlibfiles "${basepath}/mlib/utils/vector.h" )

list(APPEND mlibfiles "${basepath}/mlib/utils/string_helpers.cpp" )






list(APPEND mlibfiles "${basepath}/mlib/utils/files.cpp" )
list(APPEND mlibfiles "${basepath}/mlib/utils/mlibtime.cpp" )










ADD_LIBRARY(mlib ${mlibfiles} ) # the headers just so they show up in ide...


list(APPEND mlib-cuda-files "${basepath}/mlib/cuda/devmemmanager.h" )
list(APPEND mlib-cuda-files "${basepath}/mlib/cuda/devmemmanager.cu" )
list(APPEND mlib-cuda-files "${basepath}/mlib/cuda/devstreampool.h" )
list(APPEND mlib-cuda-files "${basepath}/mlib/cuda/devstreampool.cu" )


list(APPEND mlib-cuda-files "${basepath}/mlib/cuda/cuda_helpers.h" )
list(APPEND mlib-cuda-files "${basepath}/mlib/cuda/mbm.h" )





list(APPEND mlib-cuda-files "${basepath}/mlib/cuda/cuda_helpers.cu" )
list(APPEND mlib-cuda-files "${basepath}/mlib/cuda/mbm.cu" )

    cuda_add_library(mlib-cuda ${mlib-cuda-files})
target_link_libraries(mlib-cuda mlib ${LIBS})

list(APPEND LIBS mlib-cuda mlib)

endmacro()

