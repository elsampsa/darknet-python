#!/bin/bash
name=$(basename $PWD)
if [ "$name" == "example_module" ]
then
  echo will not overwrite default example_module directory!
  exit
fi
echo initializing project $name
# rename python module directory
pwd=$PWD
cd python/valkka
mv example_module $name
cd $pwd
# replace project names

fs="README.md CMakeLists.txt python/quicktest.py python/valkka/*/__init__.py python/valkka/README.md include/module.i.base"
for f in $fs
do
  find $f -exec sed -i "s/example_module/$name/g" {} \;
done
# decouple from git
rm -rf .git .gitignore

