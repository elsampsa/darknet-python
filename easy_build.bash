#!/bin/bash
#
# Does the standardized debian build
#
make -f debian/rules clean
make -f debian/rules build
make -f debian/rules package
