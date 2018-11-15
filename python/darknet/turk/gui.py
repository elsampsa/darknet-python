"""
gui.py : DarkTurk - A Mechanical Turk training program for Darknet
 
Copyright 2018 Valkka Security Ltd. and Sampsa Riikonen.
 
Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 
This file is part of the Darknet python bindings
 
Valkka cpp examples is free software: you can redistribute it and/or modify
it under the terms of the MIT License.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

@file    gui.py
@author  Sampsa Riikonen
@date    2018
@version 0.4.1 
@brief   DarkTurk - A Mechanical Turk training program for Darknet
"""

from PySide2 import QtWidgets, QtCore, QtGui # Qt5
# from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import os
import time
import glob
import argparse
from darknet.api2 import TrainingContext, Trainer

conf_file = "net.cfg"
"""
Neural network topology
"""

weight_file = "net.weights"
"""
Neural network weights
"""

"""
net.cfg                 : neural network topology, user

net.weights             : neural network weights, from training

names.txt               : labels, from TrainingContext

train_images/*.jpg      : training images, user

valid_images/*.jpg      : validation images, user

labels/*.txt            : boxes for each jpg file, gui user

train.txt               : image file names, from TrainingContext

valid.txt               : image file names, from TrainingContext
"""


def pathjoin(*args):
    st=""
    for a in args:
        st += a +"/"
    return st[:-1]


def QCapsulate(widget, name, blocking = False, nude = False):
    """Helper function that encapsulates QWidget into a QMainWindow
    """

    class QuickWindow(QtWidgets.QMainWindow):

        class Signals(QtCore.QObject):
            close = QtCore.Signal()

        def __init__(self, blocking = False, parent = None, nude = False):
            super().__init__(parent)
            self.propagate = True # send signals or not
            # self.setStyleSheet(style.main_gui)
            if (blocking):
                self.setWindowModality(QtCore.Qt.ApplicationModal)
            if (nude):
                # http://doc.qt.io/qt-5/qt.html#WindowType-enum
                # TODO: create a widget for a proper splashscreen (omitting X11 and centering manually)
                # self.setWindowFlags(QtCore.Qt.Popup) # Qt 5.9+ : setFlags()
                # self.setWindowFlags(QtCore.Qt.SplashScreen | QtCore.Qt.WindowStaysOnTopHint)
                self.setWindowFlags(QtCore.Qt.Dialog)
            self.signals = self.Signals()
            

        def closeEvent(self, e):
            if (self.propagate):
                self.signals.close.emit()
            e.accept()
            
        def setPropagate(self):
            self.propagate = True
            
        def unSetPropagate(self):
            self.propagate = False
            

    win = QuickWindow(blocking = blocking, nude = nude)
    win.setCentralWidget(widget)
    win.setLayout(QtWidgets.QHBoxLayout())
    win.setWindowTitle(name)
    return win



class Label:
    
    def __init__(self, index=0, center_x=0, center_y=0, width=0, height=0):
        self.index = index
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
    
    def getBox(self):
        """Get box in x, y, width, height
        """
        return (
            self.center_x - self.width/2.,
            self.center_y - self.height/2.,
            self.width,
            self.height
            )
        
    def __str__(self):
        st = ""
        # st += str(self.index) + " " +str(self.x) + " " +str(self.y) + " " +str(self.width) + " " +str(self.height)        
        b = self.getBox()
        st += str(self.index) + " "
        st += str(self.center_x) + " "
        st += str(self.center_y) + " "
        st += str(self.width) + " "
        st += str(self.height)
        return st



class Controller:
    """State of the system and widget control
    """
    
    class Signals(QtCore.QObject):
        pixmap           = QtCore.Signal(object) # current pixmap file
        labels           = QtCore.Signal(object) # list of current Label objects
        current_label    = QtCore.Signal(object) # the current Label object
        delete_file      = QtCore.Signal()
    
    
    def __init__(self, di, sub, file_list, tag_widget, tag_list, class_list):
        """
        :param file_list:   FileListContainer
        :param tag_widget:  TagWidget
        :param tag_list:    TagListContainer
        :param class_list:  ClassListContainer
        """
        self.signals=Controller.Signals()
        
        self.di = di
        self.sub = sub
        
        # state variables
        self.rect = None
        
        # signals from controller to widgets
        self.signals.pixmap.  connect(tag_widget.   set_pixmap_slot)
        self.signals.labels.  connect(tag_widget.   set_labels_slot)
        self.signals.labels.  connect(tag_list.     set_labels_slot)
        self.signals.current_label.   connect(tag_widget.   set_current_label_slot)
        
        self.signals.delete_file.connect(file_list.delete_current_file_slot)
        
        # signals from widgets to controller
        file_list.  signals.current_file.        connect(self.set_file_slot)            # new file chosen .. propagate information to all widgets
        tag_widget. signals.current_rectangle.   connect(self.set_rectangle_slot)       # a new rectangle has been created
        
        tag_list.   signals.current_tag.         connect(self.set_current_tag_slot)     # index of the current tag .. highlight corresponding rectangle
        tag_list.   signals.delete_tag.          connect(self.delete_current_tag_slot)  # removes the current chosen tag .. update everything
        
        class_list. signals.current_class.       connect(self.set_current_class_slot)   # sets current class .. that might be turned into a tag
        class_list. signals.save_class.          connect(self.save_current_class_slot)  # uses current rectangle to create a tag
        
        
    def readLabels(self, fname):
        self.data = []
        if not os.path.exists(fname):
            return
        f=open(fname,"r")
        for line in f.readlines():
            print(line.strip())
            data = line.strip().split()
            self.data.append(Label(
                index    = int(data[0]),
                center_x = float(data[1]),
                center_y = float(data[2]),
                width    = float(data[3]),
                height   = float(data[4])
                ))
        f.close()
        
        
    def set_file_slot(self, fname):
        # get the labels
        self.current_filename = fname
        
        prename = fname.split("/")[-1].split(".")[0]
        txtname = pathjoin(self.di, self.sub, prename) + ".txt"
        
        print("Controller: set_file_slot: ", fname)
        print("Controller: set_file_slot: txtname: ", txtname)
        self.readLabels(txtname)
        print("Controller: labels: ", self.data)
        
        self.txtname = txtname
        
        self.signals.pixmap.emit(fname)
        self.signals.labels.emit(self.data)
        
        
    def delete_file_slot(self):
        """Delete current file
        """
        self.signals.delete_file.emit()
        
        
    def set_rectangle_slot(self, rect):
        """:param rect: a tuple with the rectangle in relative coordinates
        """
        print("Controller: set_rectangle_slot:",rect)
        self.rect = rect
        
    def set_current_tag_slot(self, index):
        print("Controller: set_current_tag_slot:",index)
        self.current_label = self.data[index]
        print("Controller: set_current_tag_slot: self.current_label=", self.current_label)
        self.signals.current_label.emit(self.current_label)
        
    def delete_current_tag_slot(self, index):
        print("Controller: delete_current_tag_slot:",index)
        self.data.pop(index)
        self.signals.labels.emit(self.data)
        # TODO: save labels to file
        
    def set_current_class_slot(self, index):
        print("Controller: set_current_class_slot:",index)
        if self.rect:
            # save rectangle to labels
            # tell TagWidget to clear the rectangle
            self.data.append(Label(
                index    = index,
                center_x = self.rect[0],
                center_y = self.rect[1],
                width    = self.rect[2],
                height   = self.rect[3]
                ))
            
            for data in self.data:
                print(">",data)
        
            # TODO: save labels to file
            f = open(self.txtname, "w")
            for data in self.data:
                f.write(str(data)+"\n")
            f.close()
            print("set_current_class_slot: wrote file", self.txtname)
            
            self.signals.labels.emit(self.data)
            
    def save_current_class_slot(self, index):
        print("Controller: save_current_class_slot:",index)
    
    
"""
TODO: buttons:

- restart training
- continue training (check if the fil exists)


"""


class FileListContainer:
    
    class Signals(QtCore.QObject):
        current_file = QtCore.Signal(object) # informs controller about the chosen file
    
    def __init__(self, parent):
        self.widget = QtWidgets.QListWidget(parent)        
        self.widget.currentItemChanged.connect(self.item_changed_slot)
        self.signals = FileListContainer.Signals()
        self.di = "."
        self.sub = ""
        self.blocked = False # block signal emitting or not
        self.current_filename = None
        
    def readDir(self, di, sub):
        """
        :param di:  root directory
        :param sub: image directory (relative to root directory), typically "train_images"
        """
        self.blocked=True
        self.di = di
        self.sub = sub
        self.filenames=[]
        self.widget.clear()
        print("readDir:",pathjoin(self.di,self.sub,"*"))
        for fullname in glob.glob(pathjoin(self.di,self.sub,"*")):
            print(">",fullname)
            filename=fullname.split("/")[-1] 
            self.widget.addItem(QtWidgets.QListWidgetItem(filename, self.widget))
            self.filenames.append(fullname)
        self.blocked=False
        
    def write(self, fname):
        f=open(fname, "w")
        for filename in self.filenames:
            f.write(filename+"\n")
        f.close()
        
    def checkTags(self, labeldir):
        """Check that labeldir (relative to self.di root dir) exists.  If not creates it.  Checks that the directory has all necessary files
        """
        tagdir = pathjoin(self.di, labeldir)
        
        if not os.path.exists(tagdir):
            os.makedirs(tagdir)
        
        for filename in self.filenames:
            filepre = filename.split("/")[-1].split(".")[-2]
            tagfile = pathjoin(self.di, labeldir, filepre) + ".txt"
            if os.path.exists(tagfile):
                pass
            else: # create the tagging file if it does not exist
                f=open(tagfile,"w")
                f.close()
            
    # *** internal slots ***
    def item_changed_slot(self, new, old):
        self.current_filename = pathjoin(self.di, self.sub, new.text())
        if (self.blocked):
            return
        self.signals.current_file.emit(self.current_filename)
        
    def reread_dir_slot(self):
        if (self.blocked):
            return
        self.readDir(self.di, self.sub)
        
    def delete_current_file_slot(self):
        if not self.current_filename:
            return
        cr = self.widget.currentRow()
        os.remove(self.current_filename)
        self.readDir(self.di, self.sub)
        try:
            self.widget.setCurrentRow(cr)
        except:
            self.widget.setCurrentRow(0)
        
        
        


class TagWidget(QtWidgets.QWidget):
    """Widget for drawing rectangles on images
    
    X0, Y0, DX, DY are in original pixmap pixels, i.e. width: 0 .. original_width
    
    x1, x2, etc. are in screen pixmap pixels
    
    x1 = fac * X1
    
    where fac is the scaling factor which depends on how the user has scaled the widget 
    
    """
    
    class Signals(QtCore.QObject):
        current_rectangle = QtCore.Signal(object) # informs controller about the newly created rectangle
    
    # *** incoming slots ***
    def set_pixmap_slot(self, fname):
        self.reset()
        print("TagWidget : set_pixmap_slot : fname =", fname)
        self.pixmap = QtGui.QPixmap(fname)
        self.repaint()
        
    def set_labels_slot(self, rects):
        self.resetRect()
        self.data = rects
        self.current_label = None
        self.repaint()
        
    def set_current_label_slot(self, label):
        """:param label: a Label object
        """
        self.current_label = label
        self.repaint()
    
    
    def __init__(self, parent):
        super().__init__(parent)
        self.signals=TagWidget.Signals()
        self.reset()
        
    def reset(self):
        self.old_t = 0
        self.pixmap = None
        self.kx = 1
        self.ky = 1
        self.data = []
        self.current_label = None
        self.resetRect()

    def resetRect(self):
        self.rect = None
        self.X0 = None 
        
    def mousePressEvent(self, e):
        print("mousepressevent :", e)
        self.x0 = e.x()
        self.y0 = e.y()
        e.accept()
    
    def mouseReleaseEvent(self, e):
        print("mousereleaseevent :", e)
        
        self.t = time.time()
        dt = (self.t - self.old_t)
        self.old_t = self.t
        
        if (dt<0.5):
            print("double click")
            self.X0 = 0
            self.Y0 = 0
            self.DX = self.pixmap.width()
            self.DY = self.pixmap.height()
            self.repaint()
        elif self.X0 != None:
            # let's correct the coordinates
            if (self.DX < 0):
                self.X0 = self.X0+self.DX
                self.DX =- self.DX
            if (self.DY < 0):
                self.Y0 = self.Y0+self.DY
                self.DY = -self.DY
            
        
        e.accept()
        self.signals.current_rectangle.emit(self.getRect())
    
    
    def mouseMoveEvent(self, e):
        print("mousemoveevent :", e.x(), e.y())
        self.x1 = e.x()
        self.y1 = e.y()
        
        # less than image size
        self.x1 = min(self.x1, self.pixmap.width()*self.fac)
        self.y1 = min(self.y1, self.pixmap.height()*self.fac)
        
        # more than zero
        self.x1 = max(self.x1, 0)
        self.y1 = max(self.y1, 0)
        
        # more than x0
        # self.x1 = max(self.x1, self.x0)
        # self.y1 = max(self.y1, self.y0)
        
        self.X0 = self.x0 / self.fac
        self.Y0 = self.y0 / self.fac
        
        print("X0, Y0", self.X0, self.Y0)
        
        self.DX = (self.x1-self.x0) / self.fac
        self.DY = (self.y1-self.y0) / self.fac
        
        self.repaint()
        e.accept()
        
    def getDrawRect(self):
        """Return rectangle that can be draw directly
        """
        if (self.X0 != None):
            return QtCore.QRect(int(self.X0*self.fac), 
                                int(self.Y0*self.fac), 
                                int(self.DX*self.fac), 
                                int(self.DY*self.fac))
        else:
            return None
        
    def getRect(self):
        """Return rectangle in the darknet scheme (relative coordinates, center point)
        
        left upper corner = origo
        """
        w = self.pixmap.width()
        h = self.pixmap.height()
        if (self.X0 != None):
            center_x = self.X0 + self.DX/2
            center_y = self.Y0 + self.DY/2
            width    = self.DX
            height   = self.DY
            return (center_x/w, center_y/h, width/w, height/h)
        else:
            return None
        

    def paintEvent(self, e):
        print("paintevent")
        if (self.pixmap):
            qp = QtGui.QPainter()
            qp.begin(self)
            self.drawWidget(qp)
            qp.end()
            
    def resizeEvent(self, e):
        self.resetRect()
        super().resizeEvent(e)
        
        
    def sizeHint(self):
        print("default sizeHint: ", super(TagWidget, self).sizeHint())
        if (self.pixmap):
            return self.pixmap.size()
        else:
            return QtCore.QSize(300,300)
    
    
    def drawWidget(self, qp):
        size = self.size()
        
        r=(size.width()*self.pixmap.height()) / (size.height()*self.pixmap.width())
        if (r<1):       # screen form: wider than image => keep width, scale up height
            self.kx=1
            self.ky=r
        elif (r>1):  # screen form: taller than image => keep height, scale up width
            self.kx=1/r
            self.ky=1
        else:
            self.kx=1
            self.ky=1
            
        # .. with those factors, we can solve for pixmap dimensions as function of window dimensions
        
        #target = QtCore.QRectF(10.0, 20.0, 80.0, 60.0);
        #source = QtCore.QRectF(0.0, 0.0, 70.0, 40.0);
        # qp.drawPixmap(target, self.pixmap, source);
        
        w=int(self.kx*size.width())
        h=int(self.ky*size.height())
        
        # what factor has been used to scale the bitmap from its original size?
        self.fac = w / self.pixmap.width() 
        
        print("fac =", self.fac)
        
        # qp.drawPixmap(0, 0, size.width(), size.height(), self.pixmap.scaled(w, h));
        qp.drawPixmap(0, 0, self.pixmap.scaled(w, h));
        # qp.drawPixmap(0, 0, self.pixmap);
        
        print(w,h)
        
        rect = self.getDrawRect()
        
        if (rect):
            brush = QtGui.QBrush(QtGui.QColor(255,0,0,90))
            qp.setBrush(brush)
            qp.drawRect(rect)
        
        # w=self.pixmap.width(); h=self.pixmap.height();
        for d in self.data:
            
            if (self.current_label and self.current_label==d):
                pen = QtGui.QPen(QtGui.QColor(20, 255, 20), 6, QtCore.Qt.SolidLine)
            else:
                pen = QtGui.QPen(QtGui.QColor(20, 255, 20), 2, QtCore.Qt.DashLine)
            
            qp.setPen(pen)
            qp.setBrush(QtCore.Qt.NoBrush)
            
            b = d.getBox()
            rect = QtCore.QRect( # QRect starts from top-left
                int(b[0] * w),
                int(b[1] * h),
                int(b[2] * w),
                int(b[3] * h)
            )             
            qp.drawRect(rect)
            
    
            
class TagListContainer:
    """List of currently active tags
    
    /labels/*.txt
    """
    
    class Signals(QtCore.QObject):
        current_tag = QtCore.Signal(object) # informs the controller about the current active tag (class index)
        delete_tag  = QtCore.Signal(object) # tells the controller to remove a tag (and its rectangle)
    
    # slots for incoming signals
    def set_labels_slot(self, data):
        """Receives list of currently active tags
        """
        self.data = data
        self.reinit()
    
    
    def __init__(self, parent, classes):
        """
        :param classes:  list of class labels
        """
        self.widget = QtWidgets.QListWidget(parent)        
        self.widget.currentItemChanged.connect(self.item_changed_slot)
        self.widget.itemDoubleClicked.connect(self.item_double_clicked_slot)
        self.signals = TagListContainer.Signals()
        self.blocked = False # block signal emitting or not
        self.classes = classes
        self.data = [] # a list of Label objects
        self.reinit()
        
        
    def reinit(self):
        """Reinits the list widget
        """
        print("TagListContainer: reinit: clear")
        self.blocked = True
        self.widget.clear() # stupid QListWidget.clear calls triggers currentItemChanged signal
        print("TagListContainer: reinit: cleared")
        for i, data in enumerate(self.data):
            try:
                name = self.classes[data.index]
            except IndexError:
                pass
            else:
                w = QtWidgets.QListWidgetItem(name, self.widget);
                w.tag_index = i # let's add an extra attribute: tag index
                self.widget.addItem(w)
        self.blocked = False
        
    # internal slots    
    def item_changed_slot(self, new, old):
        if self.blocked: return
        print("TagListContainer: item_changed_slot", new)
        index = new.tag_index # extra attribute added to QListWidgetItem: tag index
        self.signals.current_tag.emit(index)
        
    def item_double_clicked_slot(self, item):
        if self.blocked: return
        print("TagListContainer: item_double_clicked_slot", item)
        index = item.tag_index # extra attribute added to QListWidgetItem: tag index
        self.signals.delete_tag.emit(index)
        
        
class ClassListContainer:
    
    class Signals(QtCore.QObject):
        current_class = QtCore.Signal(object) # sends controller the index of current chosen class
        save_class    = QtCore.Signal(object) # tells the controller to create a tag corresponding to this clas index
    
    def __init__(self, parent, classes):
        self.widget = QtWidgets.QListWidget(parent)        
        # self.widget.currentItemChanged.connect(self.item_changed_slot)
        self.widget.itemClicked.connect(self.item_clicked_slot)
        self.signals = ClassListContainer.Signals()
        self.classes = classes
        self.reinit()
        
    
    def reinit(self):
        """Reinits the list widget
        """
        self.widget.clear()
        for i, classname in enumerate(self.classes):
            w = QtWidgets.QListWidgetItem(classname, self.widget)
            w.setFlags(w.flags() ^ QtCore.Qt.ItemIsSelectable)
            w.class_index = i # let's add an extra attribute: tag index
            self.widget.addItem(w)
    
    def item_clicked_slot(self, new): # , old):
        index = new.class_index
        self.signals.current_class.emit(index)
    
    



class MyGui(QtWidgets.QMainWindow):

  
    def __init__(self, parent=None, directory="."):
        super(MyGui, self).__init__()
        self.di = directory
        self.initVars()
        self.setupUi()


    def initVars(self):
        pass


    def setupUi(self):
        # self.setGeometry(QtCore.QRect(100,100,500,500))

        self.w=QtWidgets.QWidget(self)    
        self.setCentralWidget(self.w)


        """

        file_list tag_widget tag_list  classes_list
        +--------+----------+-----+--------+
        |        |          |     |        |
        |        |          |     |        |  
        |        |          |     |        |
        |        |          |     |        |
        +--------+----------+-----+--------+
        
        - tag_list : list of active classes
         
        """

        self.ctx = TrainingContext.fromTemplateDir(self.di) # create a training context, based on the directory
        """
        training context does not now about directories (train_images/, valid_images/)
        .. it just reads filenames from the text files
        
        ::
        
            self.ctx.trainfile 
            self.ctx.validfile
        
        So we'll autogenerate them
        
        Class file name is in
        
        ::
        
            self.ctx.namefile
        
        """

        print(self.ctx)
        # return

        # read class names
        class_list=[]
        f=open(self.ctx.namefile)
        class_list=f.readlines()
        f.close()

        self.lay = QtWidgets.QHBoxLayout(self.w)

        self.file_list       =FileListContainer(self.w)
        self.file_list.readDir(self.di, "train_images")
        self.file_list.write(self.ctx.trainfile) # write the filelist into a text file
        self.file_list.checkTags("train_labels")

        self.lay.addWidget(self.file_list.widget)
        # self.tag_widget      = TagWidget(self.w)
        
        self.tag_widget      = TagWidget(None)
        self.tag_win = QCapsulate(self.tag_widget, "Tagging", blocking = False, nude = False)
        self.tag_win.show()
        
        self.tag_list        = TagListContainer(self.w, class_list)
        self.class_list      = ClassListContainer(self.w, class_list)
        
        self.lay.addWidget(self.file_list.widget)
        # self.lay.addWidget(self.tag_widget)
        self.lay.addWidget(self.tag_list.widget)
        self.lay.addWidget(self.class_list.widget)
        
        self.tag_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.controller = Controller(self.di, "train_labels", self.file_list, self.tag_widget, self.tag_list, self.class_list)

        # buttons
        self.buttons = QtWidgets.QWidget(self.w)
        self.lay.addWidget(self.buttons)
        self.buttons_lay = QtWidgets.QVBoxLayout(self.buttons)
        # self.train_button = QtWidgets.QPushButton("Train", self.buttons)
        self.delete_file_button = QtWidgets.QPushButton("Delete image", self.buttons)
        self.help_button = QtWidgets.QPushButton("Help!", self.buttons)
        self.buttons_lay.addWidget(self.delete_file_button)
        self.buttons_lay.addWidget(self.help_button)
        # self.buttons_lay.addWidget(self.train_button)
        # self.train_button.clicked.connect(self.train_slot)
        self.delete_file_button.clicked.connect(self.controller.delete_file_slot)
        self.help_button.clicked.connect(self.show_help)

    def train_slot(self):
        self.train(cont = False)
        

    def train(self, cont = False):
        config_file = pathjoin(self.di, conf_file)
        if cont:
            wf = pathjoin(self.di, weight_file)
        else:
            wf=""
        print("train :")
        print("train : training_ctx :\n", self.ctx, "\n")
        print("train : config_file  :", config_file)
        print("train : weight_file  :", wf)
        print("train :")
        
        # TODO: check that all these files exist: give a warning at program startup as well
        
        trainer = Trainer(
            training_ctx = self.ctx, 
            config_file = config_file,
            weight_file = wf
        )
        
        trainer()
    

    def show_help(self):
        st="""Welcome to DarkTurk(tm) !
        
Choose an image from the list
        
In the image, just keep pressing the left mouse button and drag a rectangle.  Once you are ready, choose a class from the rightmost list.

The list in the center shows active tags for the image.

To delete an active tag, double-click on it.

"Delete image" button deletes an image permanently from the directory.

That's it! :)
"""        
        QtWidgets.QMessageBox.about(self.w, "Help", st)
        
    def closeEvent(self,e):
        print("closeEvent!")
        e.accept()



def process_cl_args():
  
  def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

  parser = argparse.ArgumentParser(usage = 
    "\n" +
    sys.argv[0]+" --create_dir=true      : creates a scaffold training directory.  Read the README.md therein. \n" +
    sys.argv[0]+" --directory=$HOME/tmp  : The training directory                                              \n" +
    sys.argv[0]+" --train                : Starts training                                                     \n"
    )
  parser.register('type','bool',str2bool)
  
  parser.add_argument("--create_dir",     action="store", type=bool, default=False,      help="init scaffold directory")
  parser.add_argument("--train",          action="store", type=bool, default=False,      help="train from scratch")
  # parser.add_argument("--train_continue", action="store", type=bool, default=False,      help="continue training") # just check if file exists
  parser.add_argument("--directory",      action="store", type=str,  required=True,      help="the target scaffold directory (must exist)")
  # parser.add_argument("--n",              action="store", type=int, required=True,        help="Number of cameras to be added, for example 10")
  parsed_args, unparsed_args = parser.parse_known_args()
  return parsed_args, unparsed_args


def main():    
    parsed, unparsed = process_cl_args()
    for u in unparsed:
        print("WARNING: got unknown argument", u)
        sys.exit(3)

    if (parsed.create_dir):
        print("creating scaffold dir", parsed.directory)
        TrainingContext.makeTemplateDir(parsed.directory)
        return
        
    if (parsed.train):
        config_file = pathjoin(parsed.directory, conf_file)
        wf = pathjoin(parsed.directory, weight_file)
        if not os.path.exists(wf):
            print("Training from scratch!")
            wf=""
        else:
            print("Continuing training with", wf)

        ctx = TrainingContext.fromTemplateDir(parsed.directory)

        print("train :")
        print("train : training_ctx :\n", ctx, "\n")
        print("train : config_file  :", config_file)
        print("train : weight_file  :", wf)
        print("train :")
        
        # TODO: check that all these files exist
        
        trainer = Trainer(
            training_ctx = ctx, 
            config_file = config_file,
            weight_file = wf
        )
        trainer()
    
        
    app=QtWidgets.QApplication(["test_app"])
    mg=MyGui(directory = parsed.directory)
    mg.show()
    app.exec_()



if (__name__=="__main__"):
  main()
 
"""
TODO: 
    - cli program: create a scaffold directory structure
    - use several GPU's
    - train some shit on the GPU machine
"""

