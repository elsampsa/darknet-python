
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

    sudo dpkg -i build_dir/*.deb
    
### 3. Download parameters

With this command (woops there was an error here in earlier versions.. try again!)

    darknet_py_download
    
Or copy the directory *~/.darknet* from another installation
    
### 4. Test

With the test script "darknet_py" like this:

    darknet_py somefile.jpg
    
## Python API

Images are passed to darknet core library using numpy arrays (through memory only, of course).  Results are returned as python lists.

Python examples can be found in the *example* folder.  It's as easy as this:


    import numpy
    import time
    from PIL import Image
    from darknet.api2.predictor import get_YOLOv2_Predictor, get_YOLOv3_Predictor, get_YOLOv3_Tiny_Predictor

    filename = "dog.jpg"
    image = Image.open(filename)
    img = numpy.array(image)

    # this will take few seconds .. but we need to create predictor only once
    # predictor = get_YOLOv2_Predictor() # OK
    predictor = get_YOLOv3_Predictor() # OK
    # predictor = get_YOLOv3_Tiny_Predictor() # OK

    t = time.time()
    lis = predictor(img)
    print("Predicting took", time.time()-t, "seconds") # takes around 0.2 secs on a decent (5.1 grade) GPU
    for l in lis:
        print(l)


## Training

This python binding comes with a homebrewn tagger program, written in Python3 and Qt, called "DarkTurk (tm)" (as in mechanical turk).  Install Python Qt with this command:

    pip3 install --user --upgrade PySide2

Let's start by creating a scaffold directory structure:

    darknet_py_gui --create_dir=true --directory=$HOME/darkturk/train1

See README.md in that newly created directory.  It says:
    
    names.txt      : one object class per line
    net.cfg        : the neural network topology
    train_images/  : place your training images here
    valid_images/  : place your validation images here

So, proceed as instructed.  "names.txt" has the object names (i.e. like in file "coco.names") and net.cfg describes the neural net (like in the file "yolov3.cfg").  Start py putting your jpg images into the "train_images/" folder.  If you need image conversion to jpg, try the "darknet_py_convert" command in the "train_images/" folder.

DarkTurk(tm) will take automagically care of the rest of the necessary files.

So, now you're all set.  Start DarkTurk with:

    darknet_py_gui --directory=$HOME/darkturk/train1

You might want to press the "Help!" button for instructions

Once you have done all that tagging, close the program windows.  You can resume tagging later if you want to with that same command.

Finally, it's time to train:

    darknet_py_gui --train=true --directory=$HOME/darkturk/train1 

(you can also do the training with the canonical darknet commands as instructed in the darknet web pages)

The training can be stopped elegantly (its darknet!) from another terminal with the command:

    killall darknet_py_gui

The trained network weights appear in the "backup/" subdirectory.

For a second training iteration with more images, create a new scaffold directory:

    darknet_py_gui --create_dir=true --directory=$HOME/darkturk/train2

And copy the desired weight file from *$HOME/darkturk/train1/backup/* into *$HOME/darkturk/train2/net.weights*

Then start the trainer again, this time with 

    darknet_py_gui --train=true --directory=$HOME/darkturk/train2

Continue like this *ad nauseam*.

## Notes

The "yolov3.cfg" that comes with this package, has been modified for detection as suggested [here](https://github.com/pjreddie/darknet/issues/1104)

## License

MIT

(for darknet itself, see the **ext/darknet** folder and [darknet github](https://github.com/pjreddie/darknet))

## Authors

Sampsa Riikonen

(for darknet, see [darknet github](https://github.com/pjreddie/darknet))

## Copyright

Sampsa Riikonen 2018

(for darknet, see [darknet github](https://github.com/pjreddie/darknet))

