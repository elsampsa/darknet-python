#ifndef darknet_bridge_HEADER_GUARD
#define darknet_bridge_HEADER_GUARD
/*
 * darknet_bridge.h : Darket cpp and python interfaces to Valkka
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
 *  @file    darknet_bridge.h
 *  @author  Sampsa Riikonen
 *  @date    2018
 *  @version 0.1
 *  
 *  @brief   Darket cpp and python interfaces to Valkka
 */ 

#include "common.h"
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// #define PY_ARRAY_UNIQUE_SYMBOL shmem_array_api
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#include "darknet.h" // this header file has extern C blocks inside

extern "C" { // https://stackoverflow.com/questions/15625468/libav-linking-error-undefined-references
#include "option_list.h"
#include "utils.h"
}

image **load_alphabet_custom(const char* path); ///< darknet fix : load alphabets from a custom location

list *read_data_cfg_custom(std::string datacfg); ///< let's read the config file from a string instead

/** Returns a python list of tuples with (label, left, right, top, bottom)
 * 
 * Based on draw_detections
 * 
 */
PyObject *get_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);

/** Ripped from file detector.c
 * 
 * detector.c is not included in libdarknet.a so we must copy some stuff from that file here
 * 
 */
void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear);


/** Parameter group for the darknet cnn predictor
 * 
 */
struct DarknetContext {                       // <pyapi>
    DarknetContext(std::string datacfg,       // <pyapi>  // file with one parameter per line.  Used for training and defining the labels
                     std::string cfgfile,       // <pyapi>  // file with neural network topology
                     std::string weightfile,    // <pyapi>  // file with neural network weights
                     std::string datadir       // <pyapi>  // directory with alphabet image files
    ) : datacfg(datacfg), cfgfile(cfgfile), weightfile(weightfile), datadir(datadir) {} // <pyapi>
    std::string     datacfg;            // <pyapi> 
    std::string     cfgfile;            // <pyapi> 
    std::string     weightfile;         // <pyapi> 
    std::string     datadir;            // <pyapi>
};                                      // <pyapi>

/** The darknet cnn predictor, encapsulated in cpp and interfaced to python
 * 
 */
class DarknetPredictor {                                                                     // <pyapi>
                                                                                             // <pyapi>
public:                                                                                      // <pyapi>
    DarknetPredictor(DarknetContext ctx, float thresh = .5, float hier_thresh = .5);         // <pyapi>
    virtual ~DarknetPredictor();                                                             // <pyapi>
                                                            
protected:
    DarknetContext      ctx;
    float               thresh;
    float               hier_thresh;
    
    image               **alphabet;
    network             *net;
    char                **names;
    
    int                 height;
    int                 width;
    int                 channels;
    
    image               im;
    
protected:
    void pyArrayToImage(PyArrayObject* pyarr); ///< takes a numpy array and copies the contents to DarknetPredictor::im

public:
    PyObject* predict(PyObject* pyarr, bool draw = false);                // <pyapi>
};                                                                        // <pyapi>


class DarknetTrainer {                                                    // <pyapi>
    
public:                                                                   // <pyapi>
    DarknetTrainer(DarknetContext ctx, PyObject* py_gpu_list);            // <pyapi>
    virtual ~DarknetTrainer();                                            // <pyapi>
    
protected:
    DarknetContext      ctx;
    int     *gpus;  ///< list of gpu numbers
    int     ngpus;  ///< .. length of that list
    int     clear;  ///< what was this?
    
public:
    void train();                                                         // <pyapi>
    
    
};                                                                        // <pyapi> 



/*

/home/sampsa/C/darknet/tmp : voc.data

    classes= 20
    train  = 2012_train.txt
    valid  = 2012_val.txt
    names = voc.names
    backup = backup


/home/sampsa/C/darknet/tmp : 2012_train.txt

    /home/sampsa/C/darknet/tmp/VOCdevkit/VOC2012/JPEGImages/2011_003255.jpg
    /home/sampsa/C/darknet/tmp/VOCdevkit/VOC2012/JPEGImages/2011_003259.jpg
    /home/sampsa/C/darknet/tmp/VOCdevkit/VOC2012/JPEGImages/2011_003274.jpg
    /home/sampsa/C/darknet/tmp/VOCdevkit/VOC2012/JPEGImages/2011_003276.jpg
    
    
/home/sampsa/C/darknet/tmp/VOCdevkit/VOC2012/labels : *.txt
    

    8 0.551 0.406666666667 0.122 0.226666666667
    15 0.7 0.352 0.06 0.096
    15 0.247 0.812 0.094 0.216
    15 0.304 0.389333333333 0.224 0.245333333333
    15 0.227 0.588 0.194 0.253333333333


So it strips "JPEGImages/2011_003255.jpg" and substitutes with "labels/" ?
.. i.e. takes the last dirname away 
    

Create a directory structure:

    $BASE/[somedirectory]/*.jpg
    $BASE/labels/*.txt

Create trainfile (say, train.txt) with lines

    $BASE/[somedirectory]/[somefile.jpg]


*/

#endif

