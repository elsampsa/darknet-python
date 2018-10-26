#!/bin/bash
cp -f darknet/Makefile darknet/Makefile2
nvidia-smi
# if [[ 0 -eq 0 ]]; then # debugging
if [[ $? -eq 0 ]]; then
    echo "******************************"
    echo "*** WILL COMPILE WITH CUDA ***"
    echo "******************************"    
    sed -i -r  "s/GPU\=0/GPU\=1/g" darknet/Makefile2
    # sed -i -r  "s/CUDNN\=0/CUDNN\=1/g" darknet/Makefile2
    sed -i -r  "s/NVCC\=nvcc/NVCC\=\/usr\/local\/cuda\/bin\/nvcc/g" darknet/Makefile2
fi
echo "Makefile2 ready"
