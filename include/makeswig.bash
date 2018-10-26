#!/bin/bash
#
# extract everything that we want to expose in python

echo
echo ----------------------
echo GENERATING module.i
echo -----------------------
echo

headers="*.h"
# init module.i
cat module.i.base > module.i

for header in $headers
do
  grep -h "<pyapi>" $header | awk '{if ($1=="class" || $1=="enum" || $1=="struct") {print " "}; print $0}' >> module.i
done
