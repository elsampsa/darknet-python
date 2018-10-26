#!/bin/bash
# #
# # 1) Create a separate directory for your build (let's call it $BUILD)
# # 
# # 2) Copy this script into $BUILD
# #
# # 3) Go to $BUILD and run this script there
# #
# # (consider this: https://cmake.org/pipermail/cmake/2006-October/011711.html )
# #
# # 4) Now CMake has been configured and you can run "make" in $BUILD
# #

options=""

# # Choose either one of these:
build_type="Debug"
# build_type="Release"

# # Where to find header files?
# #
darknet_root=$HOME"/C/darknet/"

# # Substitute here your absolute path to the main dir (where you have "CMakeLists.txt")
MY_CMAKE_DIR=$HOME"/C/darknet_py/"

echo
echo $MY_CMAKE_DIR
echo
cmake $options -DCMAKE_BUILD_TYPE=$build_type -Ddarknet_root=$darknet_root $MY_CMAKE_DIR
echo
echo Run \"make\" or \"make VERBOSE=1\" to compile
echo Run \"make package\" to generate the .deb package
echo
