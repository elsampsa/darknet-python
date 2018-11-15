"""
constant.py : Constant values

* Copyright: 2018 Sampsa Riikonen
* Authors  : Sampsa Riikonen
* Date     : 2018
* Version  : 0.1

MIT LICENSE

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
from darknet.api2.tools import getDataFile, getUserFile
from darknet.api2.trainer import TrainingContext
from darknet.api2.error import WeightMissingError


def get_yolov2_weights_file():
    filename = getUserFile("yolov2.weights")
    if (os.path.exists(filename)):
        return filename
    else:
        raise(WeightMissingError("needs yolov2.weights"))


def get_yolov3_weights_file():
    filename = getUserFile("yolov3.weights")
    if (os.path.exists(filename)):
        return filename
    else:
        raise(WeightMissingError("needs yolov3.weights"))
    

def get_yolov3_tiny_weights_file():
    filename = getUserFile("yolov3-tiny.weights")
    if (os.path.exists(filename)):
        return filename
    else:
        raise(WeightMissingError("needs yolov3-tiny.weights"))


