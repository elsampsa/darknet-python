# Cmake file for Pythonized Darknet
# (C) 2018 Sampsa Riikonen
#
# cmake .
# make
# (or make VERBOSE=1)
#
#
# - Compiles and links a cpp shared library, named "darknet"
# - Generates with SWIG python bindings for the library
# - Creates a package that encapsulates the cpp library and its python bindings
# - The python part is installed as a namespace package, under namespace "darknet.core"
#

# TODO: add link-time libraries:
# pkg_search_module(CLBLAST REQUIRED clblast)
# OpenCV ? (nopes..)
# OpenCL stuff?

cmake_minimum_required(VERSION 3.1)
find_package(PkgConfig REQUIRED)

project(darknet)

# **** INPUT VARIABLES ****

## ** Necessary header files are under src/ and include/ of the main dir
SET(DARKNET_ROOT "${CMAKE_SOURCE_DIR}/ext/darknet")
option(darknet_root "darknet_root" OFF)
if    (darknet_root)
  SET(DARKNET_ROOT ${darknet_root})
endif (darknet_root)


## ** The static library appears under main dir in the pjreddie darknet version
SET(DARKNET_LIB "${DARKNET_ROOT}")
option(darknet_lib "darknet_lib" OFF)
if    (darknet_lib)
  SET(DARKNET_LIB ${darknet_lib})
endif (darknet_lib)


# ****************************************************************
# WARNING: the following three lines are modified by the "setver.bash" script
SET(MAJOR_VERSION "0") 
SET(MINOR_VERSION "2") 
SET(PATCH_VERSION "0") 
set(VERSION_STRING ${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION})

# *** define build type: Debug or Release # now from the command line
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

# ****************************************************************

# [some directory definitions]
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# [dependencies]
# exec_program(pkg-config ARGS --cflags x11 glew OUTPUT_VARIABLE GL_CFLAGS)
# exec_program(pkg-config ARGS --libs x11 glew OUTPUT_VARIABLE GL_LIBS)
pkg_search_module(PYTHON REQUIRED python3)
# populates *_INCLUDE_DIRS and *_LIBRARIES variables
pkg_search_module(OPENCL REQUIRED OpenCL)
pkg_search_module(CLBLAST REQUIRED clblast)


message("*** PYTHON INTERFACE ***")
# execute_process(COMMAND python3 -c "from distutils import sysconfig; print(sysconfig.get_python_lib(),end='')" OUTPUT_VARIABLE PYTHON_DIR)
set(PYTHON_DIR "lib/python3/dist-packages") # just hard-code it
message("PYTHON INSTALL DIR           : " ${PYTHON_DIR})

execute_process(COMMAND python3 -c "import numpy; print(numpy.get_include(),end='')" OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)
message("PYTHON NUMPY HEADER FILES IN : " ${NUMPY_INCLUDE_DIR})


# check if cuda installed
#SET(CUDAERROR 100)
#
#execute_process(
#  COMMAND nvidia-smi
#  # COMMAND ls
#  RESULT_VARIABLE CUDAERROR
#  OUTPUT_QUIET
#  ERROR_QUIET
#)
#
# SET(CUDAERROR 0) # debugging: force cuda
#
#if (NOT CUDAERROR) # eh.. this means its 0
#    message("")
#    message("*** WILL COMPILE WITH CUDA ***")
#    add_definitions("-DGPU")
#endif ()

execute_process(
  COMMAND ./makeswig.bash
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/include
)

execute_process(
  COMMAND swig -python -c++ -o ${CMAKE_SOURCE_DIR}/python/darknet/core/darknet_core_wrap.cpp -outdir ${CMAKE_SOURCE_DIR}/python/darknet/core ${CMAKE_SOURCE_DIR}/include/module.i
)

# compiler flags
# add_compile_options("-std=c++14" "-pthread") #  ${GL_CFLAGS})
# add_compile_options("-std=c++14" "-Wno-return-type") # .. -pthread required only at link time
add_compile_options("-std=c++14") # .. -pthread required only at link time

# [define library source files]
file(GLOB SOURCES src/*.cpp)
# file(GLOB SWIGBASE include/module.i.base)

# [we're compiling a library here..]
add_library(${PROJECT_NAME} SHARED ${SOURCES})
# => now the target_* commands work

# target_link_libraries(${PROJECT_NAME} "-Wl,--no-as-needed")

# [define library header files]
# include_directories(include)
target_include_directories(${PROJECT_NAME} PUBLIC include)

# [set shared library version]
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${VERSION_STRING} SOVERSION ${MAJOR_VERSION})

# [include directories]
target_include_directories(${PROJECT_NAME} PUBLIC ${NUMPY_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})
target_link_libraries(${PROJECT_NAME} "pthread")

message("DARKNET ROOT : ${DARKNET_ROOT}")

# target_include_directories(${PROJECT_NAME} PUBLIC "${VALKKA_INCLUDE_DIRS}")
# target_include_directories(${PROJECT_NAME} PUBLIC "${NUMPY_INCLUDE_DIRS}")

## ********************************************************************************
## The statically compiled library we're encapsulating here into our library, might have some dependencies on some elemental libraries
# pkg_search_module(OPENCV REQUIRED opencv)

# LDFLAGS+= `pkg-config --libs opencv` -lstdc++
# COMMON+= `pkg-config --cflags opencv` 

# ifeq ($(GPU), 1) 
# COMMON+= -DGPU -I/usr/local/cuda/include/
# CFLAGS+= -DGPU
# LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand

## header file location
# target_include_directories(${PROJECT_NAME} PUBLIC ${OPENCV_INCLUDE_DIRS})
## link time dependencies
# target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES})
## *******************************************************************************

## *************************************************************************************************
## define header file location (you have installed this by yourself)
target_include_directories(${PROJECT_NAME} PUBLIC "${DARKNET_ROOT}/include") # NOPE
target_include_directories(${PROJECT_NAME} PUBLIC "${DARKNET_ROOT}/src") # NOPE

## define where the .a files are to be found (you have compiled this by yourself)
target_link_libraries(${PROJECT_NAME} "-L${DARKNET_LIB}")
## link time dependencies
target_link_libraries(${PROJECT_NAME} "rt")
target_link_libraries(${PROJECT_NAME} "-Wl,--allow-multiple-definition" "-Wl,-Bsymbolic" "-Wl,--start-group" "-Wl,--whole-archive")
target_link_libraries(${PROJECT_NAME} ":libdarknet.a") # you could have a whole list of .a files here
target_link_libraries(${PROJECT_NAME} "-Wl,--no-whole-archive" "-Wl,--end-group")
##
## **************************************************************************************************
 
## **************************************************************************************************
## *** An example how to use a system-wide installed shared library ***
## *** Your library will depend on this one dynamically ***
#message("USING SYSTEM PROVIDED SHARED LIBRARY FOR SQLITE3")
## In this demo, our library will depend dynamically on the sqlite3 library.  Install that into your system with: sudo apt-get install libsqlite3-dev
#pkg_search_module(SQLITE REQUIRED sqlite3)
## define header file location
#target_include_directories(${PROJECT_NAME} PUBLIC ${SQLITE_INCLUDE_DIRS})
## link time dependencies
#target_link_libraries(${PROJECT_NAME} ${SQLITE_LIBRARIES})

#if (NOT CUDAERROR)
#    target_include_directories(${PROJECT_NAME} PUBLIC "/usr/local/cuda/include/")
#    target_link_libraries(${PROJECT_NAME} "-L/usr/local/cuda/lib64")
#    target_link_libraries(${PROJECT_NAME} "cuda" "cudart" "cublas" "curand")
#endif ()
target_link_libraries(${PROJECT_NAME} ${OPENCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${CLBLAST_LIBRARIES})
# **************************************************************************************************

# *** (SWIG, 3) Compile the cpp-wrapped python code ***
add_library(swig_module SHARED ${CMAKE_SOURCE_DIR}/python/darknet/core/darknet_core_wrap.cpp)
target_include_directories(swig_module PUBLIC include)
target_include_directories(swig_module PUBLIC $${NUMPY_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
target_include_directories(swig_module PUBLIC "${DARKNET_ROOT}/include") # NOPE
target_include_directories(swig_module PUBLIC "${DARKNET_ROOT}/src") # NOPE
#if (NOT CUDAERROR)
#    target_include_directories(swig_module PUBLIC "/usr/local/cuda/include/")
#    target_link_libraries(swig_module "-L/usr/local/cuda/lib64")
#    target_link_libraries(swig_module "cuda" "cudart" "cublas" "curand")
#endif ()
target_link_libraries(swig_module ${OPENCL_LIBRARIES})
target_link_libraries(swig_module ${CLBLAST_LIBRARIES})

# target_link_libraries(swig_module "-Wl,--no-as-needed")
target_link_libraries(swig_module "darknet.so")
target_link_libraries(swig_module "-L${CMAKE_CURRENT_BINARY_DIR}/lib")


set_target_properties(swig_module PROPERTIES VERSION ${VERSION_STRING} SOVERSION ${MAJOR_VERSION})
set_target_properties(swig_module PROPERTIES PREFIX "")
set_target_properties(swig_module PROPERTIES OUTPUT_NAME "_darknet_core")
set_target_properties(swig_module PROPERTIES SUFFIX ".so")
set_target_properties(swig_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/darknet/core)

add_dependencies(swig_module ${PROJECT_NAME}) # swig .so depends on the main shared library

set(TESTNAMES "mytest") # add here the names of your test binaries like this: "mytest1" "mytest2" ..
add_custom_target(tests) # Note: without 'ALL'
foreach( testname ${TESTNAMES} )
  add_executable(${testname} "test/${testname}.cpp") # Note: without 'ALL'
  target_include_directories(${testname} PUBLIC "include")
  target_include_directories(${testname} PUBLIC ${NUMPY_INCLUDE_DIR})
  target_include_directories(${testname} PUBLIC ${PYTHON_INCLUDE_DIRS})
  target_include_directories(${testname} PUBLIC "${DARKNET_ROOT}/include") # NOPE
  target_include_directories(${testname} PUBLIC "${DARKNET_ROOT}/src") # NOPE

  target_link_libraries(${testname} "${PROJECT_NAME}.so")
  target_link_libraries(${testname} "-L${CMAKE_CURRENT_BINARY_DIR}/lib")
  target_link_libraries(${testname} ${PYTHON_LIBRARIES})
  target_link_libraries(${testname} ${OPENCL_LIBRARIES})
  target_link_libraries(${testname} ${CLBLAST_LIBRARIES})

  #if (NOT CUDAERROR)
  #  target_include_directories(${testname} PUBLIC "/usr/local/cuda/include/")
  #  target_link_libraries(${testname} "-L/usr/local/cuda/lib64")
  #  target_link_libraries(${testname} "cuda" "cudart" "cublas" "curand")
  #endif ()

  

  add_dependencies(tests ${testname}) # tests depends on the executable
  add_dependencies(${testname} ${PROJECT_NAME})
endforeach( testname ${TESTNAMES} )


# *** packaging ***
# SET(CPACK_SET_DESTDIR "on") # don't use
# SET(CPACK_PACKAGING_INSTALL_PREFIX "/tmp") # don't use

exec_program(dpkg ARGS --print-architecture OUTPUT_VARIABLE MY_ARCH)

SET(CPACK_GENERATOR "DEB")
SET(CPACK_PACKAGE_NAME "python_${PROJECT_NAME}")

SET(CPACK_PACKAGE_VERSION_MAJOR "${MAJOR_VERSION}")
SET(CPACK_PACKAGE_VERSION_MINOR "${MINOR_VERSION}")
SET(CPACK_PACKAGE_VERSION_PATCH "${PATCH_VERSION}")
SET(CPACK_PACKAGE_VERSION "${VERSION_STRING}")

SET(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${MY_ARCH})

# The dependencies: keep these consistent with debian/control
# SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libpython3.5(>= 3.5.2), libgcc1(>= 1:6.0.1), libc6(>= 2.23), libgl1-mesa-glx(>= 12.0.6), libx11-6(>= 2:1.6.3), libstdc++6(>= 5.4.0), libc6(>= 2.23), libglew1.13(>= 1.13.0), python3-numpy") # ubuntu 16
# NEW: Avoid version numbers, use "utils" packages to imply a dependency, for example, to libglew (without hardcoding the version number into the package name)
# SET(CPACK_DEBIAN_PACKAGE_DEPENDS "valkka(>=0.6.0)")
SET(CPACK_DEBIAN_PACKAGE_DEPENDS "python3, python3-numpy, python3-pil")

# objdump -p libValkka.so
# dpkg -S libc.so.6
# => libc6:amd64: /lib/x86_64-linux-gnu/libc.so.6
# apt-cache show libc6 | grep "Version"
# => Version: 2.23-0ubuntu9
# a typical dependency seems to be: libc6 (>= 2.17)

SET(CPACK_PACKAGE_CONTACT "sampsa.riikonen@iki.fi")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Pythonized Darknet")
SET(CPACK_DEBIAN_PACKAGE_RECOMMENDS "")
SET(CPACK_DEBIAN_PACKAGE_SUGGESTS "")
# SET(CPACK_PACKAGE_INSTALL_DIRECTORY "dir") # don't use

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib) # install the shared library
# install(DIRECTORY "${CMAKE_SOURCE_DIR}/include" DESTINATION include/darknet FILES_MATCHING PATTERN "*.h") # install header files
# The install command: https://cmake.org/cmake/help/v3.0/command/install.html

# Cmake's INSTALL command is totally cryptic
# what the INSTALL command (maybe) does ..
# .. it takes the last bit of DIRECTORY and puts matched files into DESTINATION/last_bit

# Cmake manual:
# "The last component of each directory name is appended to the destination directory but a trailing slash may be used to avoid this because it leaves the last component empty."
# fair enough! :)

## include header files ..
# install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/" DESTINATION include/valkka_darknet FILES_MATCHING PATTERN "*.h")

## an example from Valkka's CMake file, how to include header files of used libraries:
#install(DIRECTORY "${LIVE555_ROOT}/liveMedia/include"               DESTINATION include/valkka/liveMedia             FILES_MATCHING PATTERN "*.h*")
#install(DIRECTORY "${FFMPEG_ROOT}/libavfilter"   DESTINATION include/valkka   FILES_MATCHING PATTERN "*.h")

# TODO: change from absolute to relative dir
install(DIRECTORY "${CMAKE_SOURCE_DIR}/python/darknet"  DESTINATION ${PYTHON_DIR}  FILES_MATCHING PATTERN "*.py")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/python/darknet"  DESTINATION ${PYTHON_DIR}  FILES_MATCHING PATTERN "*.so*")
# install(DIRECTORY "${CMAKE_SOURCE_DIR}/python/darknet"  DESTINATION ${PYTHON_DIR}  FILES_MATCHING PATTERN "*.png")
# install(DIRECTORY "${CMAKE_SOURCE_DIR}/python/darknet"  DESTINATION ${PYTHON_DIR}  FILES_MATCHING PATTERN "*.yuv")

install(DIRECTORY "${CMAKE_SOURCE_DIR}/python/darknet/api2/data"  DESTINATION "${PYTHON_DIR}/darknet/api2" FILES_MATCHING PATTERN "*")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/python/darknet/api2/test_data"  DESTINATION "${PYTHON_DIR}/darknet/api2"  FILES_MATCHING PATTERN "*")

install(PROGRAMS "${CMAKE_SOURCE_DIR}/bin/darknet_py"  DESTINATION "/usr/bin")
install(PROGRAMS "${CMAKE_SOURCE_DIR}/bin/darknet_py_download"  DESTINATION "/usr/bin")
install(PROGRAMS "${CMAKE_SOURCE_DIR}/bin/darknet_py_gui"  DESTINATION "/usr/bin")
install(PROGRAMS "${CMAKE_SOURCE_DIR}/bin/darknet_py_convert"  DESTINATION "/usr/bin")

# install(DIRECTORY "${FFMPEG_ROOT}" DESTINATION include/valkka   FILES_MATCHING PATTERN "*.h")

# when compiling on my linux box, I use:
# -I/home/sampsa/live555/live/UsageEnvironment/include
# -I/home/sampsa/ffmpeg/ffmpeg_git_lgpl 
#
# with cpp api, should use
# -I/usr/valkka/include/
# -I/usr/valkka/BasicUsageEnvironment/include/
# .. etc. for live555
# -I/usr/valkka/ffmpeg/
#
# however, if combined with "-I/usr/include", there might be other versions of the same header files in "/usr/include/" TODO: think about this..
#

# TODO: how to configure pkg-config when installing this .deb package?
# something like this..?
# https://gitlab.kitware.com/third-party/zlib/commit/ca6e7a0d552e3b54c0833658409e34f9de3bead6

# This must always be last!
INCLUDE(CPack)
