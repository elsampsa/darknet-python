"""
predictor.py : Higher-level encapsulation for the swig-wrapped python interface

* Copyright: 2018 Sampsa Riikonen
* Authors  : Sampsa Riikonen
* Date     : 2018
* Version  : 0.1

MIT LICENSE

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import sys
import os
import numpy
from darknet import core
from darknet.api2.tools import getDataFile, getTestDataFile, getUserFile
from darknet.api2.constant import get_yolov3_weights_file, get_yolov3_tiny_weights_file
from darknet.api2.trainer import TrainingContext

pre_mod = "darknet.api2.predictor : "



class Predictor:
    
    
    def __init__(self, training_ctx = None, config_file = None, weight_file = None, thresh = 0.9, hier_thresh = 0.5, verbose = False):
        # datacfg(datacfg), cfgfile(cfgfile), weightfile(weightfile), datadir(datadir), thresh(thresh), hier_thresh(hier_thresh)
        """
        DarknetContext ctx = DarknetContext(
        darkdir + std::string("cfg/coco.data"),     # how the weights were trained?
        darkdir + std::string("cfg/yolov3.cfg"),    # convoluted neural network scheme
        darkdir + std::string("yolov3.weights"),    # neural network weights. has been trained with the "coco" data
        darkdir + std::string("data/labels/"),      # alphabet bitmaps
        0.1, 0.5);
        """
        self.pre = pre_mod + self.__class__.__name__+" : "
        
        assert(isinstance(training_ctx, TrainingContext))
        assert(isinstance(config_file, str))
        assert(isinstance(weight_file, str))
        assert(isinstance(thresh, float))
        assert(isinstance(hier_thresh, float))
        
        assert(os.path.exists(config_file))
        assert(os.path.exists(weight_file))
        
        self.verbose = verbose
        
        if (self.verbose):
            print(self.pre)
            print(self.pre, str(training_ctx))
            print(self.pre)
            print(self.pre, config_file)
            print(self.pre, weight_file)
            print(self.pre, getDataFile("labels/"))
            
        self.thresh = thresh
        self.hier_thresh = hier_thresh
        
        # return
        
        self.ctx = core.DarknetContext(
            str(training_ctx),
            config_file,
            weight_file,
            getDataFile("labels/")
            )
        
        self.predictor = core.DarknetPredictor(self.ctx, thresh, hier_thresh);
        
            
    def __call__(self, img):
        assert(isinstance(img, numpy.ndarray)) # must be a numpy array
        assert(img.dtype == numpy.uint8) # must be bytes
        assert(len(img.shape)==3) # must conform to height, width, number of channels
        
        # dic = {} # {"object_name": (probability, (left, right, top, bottom))}
        # lis = [ ("eka", 0.1, 1, 2, 3, 4), ("toka", 0.2, 11, 22, 33, 44), ("eka", 0.3, 21, 22, 23, 24) ] # debug
        lis = self.predictor.predict(img) # list element: ("object name", probability, left, right, top, bottom)
        
        """ # not a good idea ..
        for name, prob, left, right, top, bottom in lis:
            if (name not in dic or prob > dic[name][0]):
                dic[name] = (prob, (left, right, top, bottom))
        """
        # just return list as is
        return lis
            
            

def get_YOLOv3_Predictor():
    # has been trained with coco
    file_yolov3_cfg = getDataFile("yolov3/yolov3.cfg")
    
    yolov3_training_ctx = TrainingContext(
        n_classes = 80,
        trainfile = "/home/pjreddie/data/coco/trainvalno5k.txt",
        validfile = "coco_testdev",
        namefile = getDataFile("yolov3/coco.names"), # all the other values don't make any difference when predicting
        backup_dir ="/home/pjreddie/backup/",
        setname = "coco"
    )

    return Predictor(training_ctx = yolov3_training_ctx, weight_file = get_yolov3_weights_file(), config_file = file_yolov3_cfg)
    

def get_YOLOv3_Tiny_Predictor():
    # has been trained with coco
    file_yolov3_tiny_cfg = getDataFile("yolov3-tiny/yolov3-tiny.cfg")
    
    yolov3_tiny_training_ctx = TrainingContext(
        n_classes = 80,
        trainfile = "/home/pjreddie/data/coco/trainvalno5k.txt",
        validfile = "coco_testdev",
        namefile = getDataFile("yolov3-tiny/coco.names"), # all the other values don't make any difference when predicting
        backup_dir ="/home/pjreddie/backup/",
        setname = "coco"
    )
    
    return Predictor(training_ctx = yolov3_tiny_training_ctx, weight_file = get_yolov3_tiny_weights_file(), config_file = file_yolov3_tiny_cfg)
        


def test1():
    st="""Empty test
    """
    pre=pre_mod+"test1 :"
    print(pre,st)

    ctx = TrainingContext(
        n_classes = 80,
        trainfile = "/home/pjreddie/data/coco/trainvalno5k.txt",
        validfile = "coco_testdev",
        namefile = "/home/sampsa/C/darknet/data/coco.names",
        backup_dir ="/home/pjreddie/backup/",
        setname = "coco"
    )

    predictor = Predictor(training_ctx = ctx, weight_file = "/home/sampsa/C/darknet/yolov3.weights", config_file = "/home/sampsa/C/darknet/cfg/yolov3.cfg")

  
def test2():
    st="""Empty test
    """
    pre=pre_mod+"test2 :"
    print(pre,st)

    ctx = TrainingContext(
        n_classes = 80,
        trainfile = "/home/pjreddie/data/coco/trainvalno5k.txt",
        validfile = "coco_testdev",
        namefile = "/home/sampsa/C/darknet/data/coco.names",
        backup_dir ="/home/pjreddie/backup/",
        setname = "coco"
    )

    predictor = Predictor(training_ctx = ctx, weight_file = "/home/sampsa/C/darknet/yolov3.weights", config_file = "/home/sampsa/C/darknet/cfg/yolov3.cfg")

    a = numpy.zeros((1080, 1920, 3), dtype=numpy.uint8)

    predictor(a)


def test3():
    st="""Empty test
    """
    pre=pre_mod+"test3 :"
    print(pre,st)

    predictor = get_YOLOv3_Predictor()
    # predictor = get_YOLOv3_Tiny_Predictor()
    
    a = numpy.zeros((1080, 1920, 3), dtype=numpy.uint8)

    predictor(a)


def test4():
    st="""Empty test
    """
    pre=pre_mod+"test4 :"
    print(pre,st)

    from PIL import Image

    filename = getTestDataFile("dog.jpg")
    image = Image.open(filename)
    img = numpy.array(image)
    
    # predictor = get_YOLOv3_Predictor()
    predictor = get_YOLOv3_Tiny_Predictor()
    
    lis = predictor(img)
    print(pre,lis)
    for l in lis:
        print(pre,l)



def main():
  pre=pre_mod+"main :"
  print(pre,"main: arguments: ",sys.argv)
  if (len(sys.argv)<2):
    print(pre,"main: needs test number")
  else:
    st="test"+str(sys.argv[1])+"()"
    exec(st)
  
  
if (__name__=="__main__"):
  main()

