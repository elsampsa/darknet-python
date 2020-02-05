#!/bin/bash
#
# Does the standardized debian build
#
cd ext/CLBlast
make -f debian/rules clean
make -f debian/rules build
make -f debian/rules binary
cd ../..
