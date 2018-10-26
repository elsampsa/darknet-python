"""
NAME.py : Description of the file

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
from darknet import core
from darknet.api2.tools import getDataFile

pre_mod = "darknet.api2.trainer : "


def makeDirIf(di):
    try:
        os.makedirs(di)
    except FileExistsError:
        pass


class TrainingContext:
    """
    
    ::
    
        classes= 80                                              # n_classes
        train  = /home/pjreddie/data/coco/trainvalno5k.txt       # trainfile
        valid  = coco_testdev                                    # validfile
        names = /home/sampsa/C/darknet/data/coco.names           # namefile
        backup = /home/pjreddie/backup/                          # backup_dir
        eval=coco                                                # setname
        
      
    n_classes : Number of object classes
      
    namefile : Object class names.  One per line.  Number of lines = n_classes
        
    trainfile : (see below)
    
    **Original Darknet directory structure**
    
    The *namefile* with object label names can be anywhere
    
    For training, the following directory structure for *image files* and *object files* is mandatory:

    ::
      
        $BASE/[somedirectory]/*.jpg
        $BASE/labels/*.txt

    There each .txt *objectfile* has multiple lines like this:
    
    ::
    
        <object-class> <x> <y> <width> <height>


    *trainfile* can be located anywhere.  It must have absolute paths to images, one in each line:
    
    ::

        $BASE/[somedirectory]/[somefile.jpg]
        
      
    **Directory structure used here**
      
    Let's use the following scheme:
    
    ::
        
        $BASE/
            README          : some info
            names.txt       : namefile
            train.txt       : trainfile
            valid.txt       : validfile
            images/         : image files
            labels/         : object files
            backup/         : backup dir (?)
    
    
        
    """
    
    @staticmethod
    def makeTemplateDir(directory):
        if (os.path.exists(directory)):
            print("TrainingContext : warning! directory", directory, "exists")
        else:
            os.makedirs(directory)
        
        
        makeDirIf(os.path.join(directory, "images"))
        makeDirIf(os.path.join(directory, "labels"))
        makeDirIf(os.path.join(directory, "backup"))
        
        readmefile = os.path.join(directory, "README")
        namefile = os.path.join(directory, "names.txt")
        trainfile = os.path.join(directory, "train.txt")
        validfile = os.path.join(directory, "valid.txt")
        
        # if not os.path.exists(readmefile):
        f=open(readmefile, "w")
        f.writelines([
            "\n",
            "names.txt : lines should look like this:\n",
            "objectname1\n",
            "objectname2\n",
            "\n",
            "train.txt : lines should look like this:\n",
            directory+"/images/objectimage1.jpg\n",
            directory+"/images/objectimage2.jpg\n",
            "\n",
            "valid.txt : same format as train.txt\n",
            "\n",
            "you should have files like this for each image:\n",
            "labels/objectimage1.txt\n",
            "labels/objectimage1.txt\n",
            "\n",
            "they have lines like this:\n",
            "objectname x y width height\n",
            "\n"
            ])
        f.close()
    
        if not os.path.exists(namefile):
            f=open(namefile, "w")
            f.write("")
            f.close()
        
        if not os.path.exists(trainfile):
            f=open(trainfile, "w")
            f.write("")
            f.close()
            
        if not os.path.exists(validfile):
            f=open(validfile, "w")
            f.write("")
            f.close()
        
        
    @staticmethod
    def fromTemplateDir(directory):
        namefile = os.path.join(directory, "names.txt")
        trainfile = os.path.join(directory, "train.txt")
        validfile = os.path.join(directory, "valid.txt")
        backup_dir = os.path.join(directory, "backup")
        
        f = open(namefile, "r")
        n_classes = len(f.readlines())
        f.close()
        
        return TrainingContext(namefile = namefile, trainfile = trainfile, validfile = validfile, backup_dir = backup_dir, n_classes = n_classes)
        
        
        
    def __init__(self, validfile = None, setname = "nada", n_classes = None, trainfile = None, namefile = None, backup_dir = None):
        isinstance(validfile, str)
        isinstance(setname, str)
        isinstance(n_classes, int)
        isinstance(trainfile, str)
        isinstance(namefile, str)
        isinstance(backup_dir, str)
        
        # assert(os.path.exists(trainfile))
        assert(os.path.exists(namefile))
        # assert(os.path.exists(backup_dir))
        
        self.validfile = validfile
        self.setname = setname
        self.n_classes = n_classes
        self.trainfile = trainfile
        self.namefile = namefile
        self.backup_dir = backup_dir
        
    def __str__(self):
        """Don't touch this!  It's given to the darknet core c library as-is
        """
        st = ""
        st += "classes = "+str(self.n_classes)+"\n"
        st += "train = "+str(self.trainfile)+"\n"
        st += "valid = "+str(self.validfile)+"\n"
        st += "names = "+str(self.namefile)+"\n"
        st += "backup = "+str(self.backup_dir)+"\n"
        st += "eval = "+str(self.setname)+"\n"
        return st
    
    

class Trainer:
    
    def __init__(self, training_ctx = None, config_file = None, weight_file = ""):
        # TODO: pretrained network ?
        
        self.ctx = core.DarknetContext(
            str(training_ctx),
            config_file,
            weight_file,
            getDataFile("labels/")
            )
        
        self.trainer = core.DarknetTrainer(self.ctx, [])
        
    
    def __call__(self):
        self.trainer.train()
        
        
def test1():
    st="""Empty test
    """
    pre=pre_mod+"test1 :"
    print(pre,st)
  
    di = "/home/sampsa/tmp/darknet_test"
  
    TrainingContext.makeTemplateDir(di) # create a template for training
    # edit / insert your files
    ctx = TrainingContext.fromTemplateDir(di) # create a training context, based on the directory
    
    # print(ctx)
    
    trainer = Trainer(
        training_ctx = ctx, 
        # config_file = os.path.join(di,"mynet.cfg"),
        config_file = os.path.join(di,"yolov3-voc.cfg"),
        # weight_file = os.path.join(di, "mynet.weights")
        # weight_file = os.path.join(di, "darknet53.conv.74")
        )
    
    # do the train
    trainer()


def test2():
  st="""Empty test
  """
  pre=pre_mod+"test2 :"
  print(pre,st)
  

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

