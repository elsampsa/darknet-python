"""
NAME.py :
 
Copyright 2018 Valkka Security Ltd. and Sampsa Riikonen.
 
Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 
This file is part of the Darknet python bindings
 
Valkka cpp examples is free software: you can redistribute it and/or modify
it under the terms of the MIT License.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

@file    NAME.py
@author  Sampsa Riikonen
@date    2018
@version 0.4.1 
@brief   
"""

from PySide2 import QtWidgets, QtCore, QtGui # Qt5
import sys
from valkka.core import *


class MyGui(QtWidgets.QMainWindow):

  
  def __init__(self,parent=None):
    super(MyGui, self).__init__()
    self.initVars()
    self.setupUi()
    self.openValkka()
    

  def initVars(self):
    pass


  def setupUi(self):
    self.setGeometry(QtCore.QRect(100,100,500,500))
    
    self.w=QtWidgets.QWidget(self)
    self.setCentralWidget(self.w)
    
    
  def openValkka(self):
    pass
    
  
  def closeValkka(self):
    pass
  
  
  def closeEvent(self,e):
    print("closeEvent!")
    self.closeValkka()
    e.accept()



def main():
  app=QtWidgets.QApplication(["test_app"])
  mg=MyGui()
  mg.show()
  app.exec_()



if (__name__=="__main__"):
  main()
 
