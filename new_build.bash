#!/bin/bash
if [ $# -ne 1 ]; then
  echo "Give a build name!"
  exit
fi
dirname="build_"$1
echo "Your build is in "$dirname
mkdir -p $dirname
cp tools/build/* $dirname
