
# Python3 API for Darknet

*This is a Python3 API for pjreddie's original darknet/yolo cpp code.  For running neural-net based
object detection in python, nowadays you might prefer pytorch or tensorflow-based approached instead.
A nice alternative is, for example, the* [Detectron framework](https://github.com/facebookresearch/detectron2).

Darknet is an OpenSource convoluted neural network library, see [here](https://pjreddie.com/darknet/)

This package features a Python3 API for darknet together with debian packaging for easy distribution.

*No config files, image label jpg directories (!) etc.  Just a clean & simple Python3 API that works out of the box*


## Changelog

- version 0.2.3

    - Nasty memleak fixed, related to [this](https://stackoverflow.com/questions/3512414/does-this-pylist-appendlist-py-buildvalue-leak) : should use PyList/PyTuple_SetItem to steal the references.

- version 0.2.2

    - Added some options to the ```darknet_py``` test command

- version 0.2.1

    Fixed a memleak

    - The Predictor dtor was unable to free the cuda memory with ```free_network```.
    - ..so added a call to ```cudaDeviceReset```.  Now there's no stray memory left on the GPU.

- version 0.2.0

## Getting started

### 0. Install CUDA

This step is not obligatory.  Just do it if you want to use GPUs.

It's recommended to use a GPU with "compute capability" >= 5, see the scores [here](https://developer.nvidia.com/cuda-gpus).

Latest nvidia drivers and cuda can be found from [nvidia](https://developer.nvidia.com/cuda-downloads)

The recommended method is the "debian (local)" one.  Before installing, you should remove an earlier installation of nvidia graphic drivers (say, use 'sudo apt-get remove --purge nvidia*')

That package from nvidia will install you a recent graphics card driver and cuda that are compatible with each other.
*This will overwrite the nvidia driver that comes with your standard package management and updates*

It may happen that when you do an update in the future with apt, it will overwrite the package you have installed here.  In that case, just do 'sudo apt-get remove --purge nvidia*' and
install drivers directly from nvidia again.

(to disable ubuntu auto updates, search at launch "software & updates" => go to "updates" tab and disable updates to "never")

After installing the package, log out & in and then see your GPU's specs using the command 'nvidia-smi'.


<!-- For ubuntu 18, you may even want to try your luck with this automagic installation script

    bootstrap/darknet_py_ubuntu18_cuda_install -->

### 1. Compile

You need at least:

    sudo apt-get install build-essential cmake pkg-config swig python3 python3-dev python3-numpy python3-pil

To compile darknet with python bindings and to create a debian package for installing, just run

    ./easy_build.bash
    
It should detect the CUDA installation automagically if present
    
If you get this error:
```
cuda_runtime.h: No such file or directory
```

You must manually create a link to the correct cuda version, i.e.
```
sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda
```

### 2. Install

The debian package with

    sudo dpkg -i build_dir/*.deb
    
### 3. Download parameters

With this command:

    darknet_py_download
    
Or copy the directory *~/.darknet* from another installation
    
### 4. Test

A quicktest:

    cd example
    python3 predict_2.py
    
If everything is OK, the program should report something like this:

    Predicting took 0.03300929069519043 seconds
    
Test with any image using the script "darknet_py" like this:

    darknet_py <yolo_flavor> somefile.jpg
    
where ```<yolo_flavor>``` can be v3, v2 or v3tiny
    
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

    # each element of the list is a tuple with:
    # (class_name, probability_in_%, left, right, top, bottom)
    # in another words:
    # (x0, y0) = (left, bottom)
    # (x1, y1) = (right, top)
    # in y, smallest value corresponds to bottom left corner in the image, so if you use with cv2 / numpy, remember to invert the y axis

## GPU memory requirements

Approximately, as reported with nvidia-smi
```
Yolo v3          2400 MB
Tiny Yolo v3     230 MB
Yolo v2          1230 MB
```
A tip: if you want to save GPU memory, disable your desktop compositor.


## Video Surveillance with Yolo

If you need video surveillance with object detection, that works out-of-the-box, try [this](https://elsampsa.github.io/valkka-live/) program
        
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

## TODO

Need to migrate this to alexbe darknet asap.  Any volunteers?

## License

MIT

(for darknet itself, see the **ext/darknet** folder and [darknet github](https://github.com/pjreddie/darknet))

## Authors

Sampsa Riikonen

(for darknet, see [darknet github](https://github.com/pjreddie/darknet))

## Copyright

Sampsa Riikonen 2018

(for darknet, see [darknet github](https://github.com/pjreddie/darknet))

