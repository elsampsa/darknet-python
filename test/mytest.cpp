/*
 * mytest.cpp : test your valkka module code at the cpp level
 * 
* Copyright 2018 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of Valkka cpp examples
 * 
 * Valkka cpp examples is free software: you can redistribute it and/or modify
 * it under the terms of the MIT License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

/** 
 *  @file    mytest.cpp
 *  @author  Sampsa Riikonen
 *  @date    2018
 *  @version 0.2.1 
 *  
 *  @brief   test your valkka module code at the cpp level
 *
 */ 

#include "darknet_bridge.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
    const char* name = "@TEST: mytest: test 1: instantiate Predictor";
    std::cout << name <<"** @@DESCRIPTION **" << std::endl;
    
    // MyFrameFilter framefilter();
    
    // test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, .5, outfile, fullscreen);
    // test_detector("cfg/coco.data", "eka", "toka", "kolkki", .8, .5, "nelkki", 1);
    
    std::string darkdir = std::string("/home/sampsa/C/darknet/");
    
    DarknetContext ctx = DarknetContext(
        // darkdir + std::string("cfg/coco.data"),
        "classes= 80\ntrain  = /home/pjreddie/data/coco/trainvalno5k.txt\nvalid  = coco_testdev\n#valid = data/coco_val_5k.list\nnames = /home/sampsa/C/darknet/data/coco.names\nbackup = /home/pjreddie/backup/\neval=coco\n",
        darkdir + std::string("cfg/yolov3.cfg"),
        darkdir + std::string("yolov3.weights"),
        darkdir + std::string("data/labels/"));
                         
    npy_intp dims[] = { 1080, 1920, 3 };
    PyObject *array;
    
    Py_Initialize();
    
    // stupid fix for the numpy bug
    
    #define import_array() {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); } }

    import_array();
    
    array = PyArray_SimpleNew(3, dims, NPY_BYTE);
    PyObject_Print(array, stdout, 0);
    
    DarknetPredictor dp = DarknetPredictor(ctx, .5, .5);  
    
    std::cout << "\nPredict\n";
    dp.predict(array);
    
    std::cout <<"\nDecref\n";
    Py_DECREF(array);
    
    std::cout <<"\nBye\n";
}


void test_2() {
    const char* name = "@TEST: mytest: test 2: GPU-compiled without GPU";
    std::cout << name <<"** @@DESCRIPTION **" << std::endl;

    // MyFrameFilter framefilter();
    
    // test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, .5, outfile, fullscreen);
    // test_detector("cfg/coco.data", "eka", "toka", "kolkki", .8, .5, "nelkki", 1);
    
    std::string darkdir = std::string("/home/sampsa/C/darknet/");
  
    gpu_index = -1; // this should do the trick?
    
    DarknetContext ctx = DarknetContext(
        // darkdir + std::string("cfg/coco.data"),
        "classes= 80\ntrain  = /home/pjreddie/data/coco/trainvalno5k.txt\nvalid  = coco_testdev\n#valid = data/coco_val_5k.list\nnames = /home/sampsa/C/darknet/data/coco.names\nbackup = /home/pjreddie/backup/\neval=coco\n",
        darkdir + std::string("cfg/yolov3.cfg"),
        darkdir + std::string("yolov3.weights"),
        darkdir + std::string("data/labels/"));
                         
    npy_intp dims[] = { 1080, 1920, 3 };
    PyObject *array;
    
    Py_Initialize();
    
    // stupid fix for the numpy bug
    
    #define import_array() {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); } }

    import_array();
    
    array = PyArray_SimpleNew(3, dims, NPY_BYTE);
    PyObject_Print(array, stdout, 0);
    
    DarknetPredictor dp = DarknetPredictor(ctx, .5, .5);  
    
    std::cout << "\nPredict\n";
    dp.predict(array);
    
    std::cout <<"\nDecref\n";
    Py_DECREF(array);
    
    std::cout <<"\nBye\n";
  
  
  
  
}


void test_3() {
  
  const char* name = "@TEST: mytest: test 3: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}


void test_4() {
  
  const char* name = "@TEST: mytest: test 4: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}


void test_5() {
  
  const char* name = "@TEST: mytest: test 5: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}



int main(int argc, char** argcv) {
  if (argc<2) {
    std::cout << argcv[0] << " needs an integer argument.  Second interger argument (optional) is verbosity" << std::endl;
  }
  else {
    
    if  (argc>2) { // choose verbosity
      switch (atoi(argcv[2])) {
        case(0): // shut up
          // ffmpeg_av_log_set_level(0);
          // fatal_log_all();
          break;
        case(1): // normal
          break;
        case(2): // more verbose
          //  ffmpeg_av_log_set_level(100);
          //  debug_log_all();
          break;
        case(3): // extremely verbose
          // ffmpeg_av_log_set_level(100);
          // crazy_log_all();
          break;
        default:
          std::cout << "Unknown verbosity level "<< atoi(argcv[2]) <<std::endl;
          exit(1);
          break;
      }
    }
    
    switch (atoi(argcv[1])) { // choose test
      case(1):
        test_1();
        break;
      case(2):
        test_2();
        break;
      case(3):
        test_3();
        break;
      case(4):
        test_4();
        break;
      case(5):
        test_5();
        break;
      default:
        std::cout << "No such test "<<argcv[1]<<" for "<<argcv[0]<<std::endl;
    }
  }
} 

/*  Some useful code:


if (!stream_1) {
  std::cout << name <<"ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1"<< std::endl;
  exit(2);
}
std::cout << name <<"** test rtsp stream 1: "<< stream_1 << std::endl;
  
if (!stream_2) {
  std::cout << name <<"ERROR: missing test stream 2: set environment variable VALKKA_TEST_RTSP_2"<< std::endl;
  exit(2);
}
std::cout << name <<"** test rtsp stream 2: "<< stream_2 << std::endl;
    
if (!stream_sdp) {
  std::cout << name <<"ERROR: missing test sdp file: set environment variable VALKKA_TEST_SDP"<< std::endl;
  exit(2);
}
std::cout << name <<"** test sdp file: "<< stream_sdp << std::endl;

  
*/


