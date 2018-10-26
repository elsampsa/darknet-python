
# Python3 bindings for Darknet

Darknet is an OpenSource convoluted neural network library, see [here](https://pjreddie.com/darknet/)

Here we create a neat debian package for darknet with python3 bindings.  

## Getting started

### 0. Install CUDA

from [nvidia](https://developer.nvidia.com/cuda-downloads)

if you have a GPU.  It's recommended to use a GPU with "compute capability" >= 5, see the scores [here](https://developer.nvidia.com/cuda-gpus).

To see your GPU's specs, use the command 'nvidia-smi'.

The method that best worked for me, is the "debian (local)" one.  Before installing, you should remove an earlier installation of nvidia graphic drivers (say, use 'sudo apt-get remove --purge nvidia-*')

For ubuntu 18, you may even want to try your luck with this automagic installation script

    bootstrap/darknet_py_ubuntu18_cuda_install

### 1. Compile

You need at least:

    sudo apt-get install build-essential libc6-dev cmake pkg-config swig libstdc++-5-dev python3 python3-dev python3-numpy python3-pil


To compile darknet with python bindings and to create a debian package for installing, just run

    ./easy_build.bash
    
It should detect the CUDA installation automagically if present
    
### 2. Install

The debian package with

    sudo dpkg -i build_dir/python_darknet-0.1.0-Linux.deb
    
### 3. Download parameters

With this command

    darknet_py_download
    
Or copy the directory *~/.darknet* from another installation
    
### 4. Test

With the test script "darknet_py" like this:

    darknet_py somefile.jpg
    
## Python API

Images are passed to darknet core library using numpy arrays (through memory only, of course).  Results are returned as python lists.

Examples can be found in the *example* folder.

## Notes

The "yolov3.cfg" that comes with this package, has been hacked as suggested [here](https://github.com/pjreddie/darknet/issues/1104)

## License

MIT

(for darknet itself, see the **ext/darknet** folder and [darknet github](https://github.com/pjreddie/darknet))

## Authors

Sampsa Riikonen

(for darknet, see [darknet github](https://github.com/pjreddie/darknet))

## Copyright

Sampsa Riikonen 2018

(for darknet, see [darknet github](https://github.com/pjreddie/darknet))

