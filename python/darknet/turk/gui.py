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
import glob


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
        st += str(b[0]) + " "
        st += str(b[1]) + " "
        st += str(b[2]) + " "
        st += str(b[3]) + " "
        return st




class TagWidget(QtWidgets.QWidget):
    """Widget for drawing rectangles on images
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        print("__init__")
        self.reset()
        
    def reset(self):
        self.pixmap = None
        self.rect = None
        self.kx = 1
        self.ky = 1
        self.X0 = None
        self.data = []
    
    def mousePressEvent(self, e):
        print("mousepressevent :", e)
        self.x0 = e.x()
        self.y0 = e.y()
        e.accept()
    
    def mouseReleaseEvent(self, e):
        print("mousereleaseevent :", e)
        e.accept()
    
    
    def mouseMoveEvent(self, e):
        print("mousemoveevent :", e.x(), e.y())
        self.x1 = e.x()
        self.y1 = e.y()
        
        self.X0 = self.x0 / self.fw
        self.Y0 = self.y0 / self.fh
        
        print("X0, Y0", self.X0, self.Y0)
        
        self.DX = (self.x1-self.x0) / self.fw
        self.DY = (self.y1-self.y0) / self.fh
        
        self.repaint()
        e.accept()
        
    def getDrawRect(self):
        """Return rectangle that can be draw directly
        """
        if (self.X0):
            return QtCore.QRect(int(self.X0*self.fw), 
                                int(self.Y0*self.fh), 
                                int(self.DX*self.fw), 
                                int(self.DY*self.fh))
        else:
            return None
    
    
    def paintEvent(self, e):
        print("paintevent")
        if (self.pixmap):
            qp = QtGui.QPainter()
            qp.begin(self)
            self.drawWidget(qp)
            qp.end()
            
        
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
        
        # what factors have been used to scale the bitmap from its original size?
        self.fw = w / self.pixmap.width() 
        self.fh = h / self.pixmap.height()
        
        print("r=",r)
        
        # qp.drawPixmap(0, 0, size.width(), size.height(), self.pixmap.scaled(w, h));
        qp.drawPixmap(0, 0, self.pixmap.scaled(w, h));
        # qp.drawPixmap(0, 0, self.pixmap);
        
        print(w,h)
        
        rect = self.getDrawRect()
        
        if (rect):
            brush = QtGui.QBrush(QtGui.QColor(255,0,0,90))
            qp.setBrush(brush)
            qp.drawRect(rect)
        
        pen = QtGui.QPen(QtGui.QColor(20, 20, 20), 4, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.setBrush(QtCore.Qt.NoBrush)
        
        # w=self.pixmap.width(); h=self.pixmap.height();
        for d in self.data:
            b = d.getBox()
            rect = QtCore.QRect( # QRect starts from top-left
                int(b[0] * w),
                int(b[1] * h),
                int(b[2] * w),
                int(b[3] * h)
            )             
            qp.drawRect(rect)
            
        
    def set_file_slot(self, fname):
        self.reset()
        print("TagWidget : set_file_slot : fname =", fname)
        self.pixmap = QtGui.QPixmap(fname)
        self.repaint()
        
        # get the rectangles
        txtname=fname.split("/")[-1].split(".")[0] # last part of the path, take name from "name.txt"
        
        path=""
        for part in fname.split("/")[0:-2]:
            # print(part, path)
            path += part+"/"
        
        fname = path + "labels/" + txtname + ".txt"
        print(fname)
        
        self.rect_filename = fname
        
        self.readRects()
        
        self.repaint()
        # self.resize(self.pixmap.size())
        

    def readRects(self):
        self.data = []
        if not os.path.exists(self.rect_filename):
            return
        f=open(self.rect_filename,"r")
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
        



class ListContainer:
    
    class Signals(QtCore.QObject):
        current_file = QtCore.Signal(object)
    
    def __init__(self):
        self.widget = QtWidgets.QListWidget()        
        self.widget.currentItemChanged.connect(self.item_changed_slot)
        self.signals = ListContainer.Signals()
        self.di = "."
        
    def readDir(self, di):
        self.di = di
        self.widget.clear()
        for fullname in glob.glob(os.path.join(self.di,"*")):
            filename=fullname.split("/")[-1] 
            self.widget.addItem(QtWidgets.QListWidgetItem(filename, self.widget))
        
        
    def item_changed_slot(self, new, old):
        self.signals.current_file.emit(os.path.join(self.di, new.text()))
        
        



class MyGui(QtWidgets.QMainWindow):

  
    def __init__(self,parent=None):
        super(MyGui, self).__init__()
        self.initVars()
        self.setupUi()
        self.makeLogic()


    def initVars(self):
        pass


    def setupUi(self):
        # self.setGeometry(QtCore.QRect(100,100,500,500))

        self.w=QtWidgets.QWidget(self)    
        self.setCentralWidget(self.w)


        """

        files  image   classes   action buttons
        +---+----------+-----+
        |   |          |     |   [clear]
        |   |          |     |   [save]
        |   |          |     |      
        |   |          |     |   [train]
        +---+----------+-----+

        - active classes are highlighted
        - pressing class shows associated boxes
        - 
        """

        self.lay = QtWidgets.QHBoxLayout(self.w)

        self.file_list       =ListContainer()
        self.file_list.widget.setParent(self.w)
        self.file_list.readDir("/home/sampsa/tmp/darknet_test/images/")

        self.lay.addWidget(self.file_list.widget)
        self.image           = TagWidget(self.w)
        self.classes_list    = QtWidgets.QListWidget(self.w)
        self.buttons         = QtWidgets.QWidget(self.w)
        self.buttons_lay     = QtWidgets.QHBoxLayout(self.buttons)

        self.lay.addWidget(self.file_list.widget)
        self.lay.addWidget(self.image)
        self.lay.addWidget(self.classes_list)
        self.lay.addWidget(self.buttons)

        self.save_button = QtWidgets.QPushButton("SAVE", self.buttons)
        self.buttons_lay.addWidget(self.save_button)

        self.image.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)


    def makeLogic(self):
        self.file_list.signals.current_file.connect(self.image.set_file_slot)
        


    def closeEvent(self,e):
        print("closeEvent!")
        e.accept()



def main():
    app=QtWidgets.QApplication(["test_app"])
    mg=MyGui()
    mg.show()
    app.exec_()



if (__name__=="__main__"):
  main()
 
