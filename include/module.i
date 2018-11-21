%module darknet_core
%include <std_string.i>
%include "cpointer.i" // simple pointer types for c(pp).  We use them for pass-by-reference cases
/* Create some functions for working with "int *" */
%pointer_functions(int, intp);

%{ // this is prepended in the wapper-generated c(pp) file
#define SWIG_FILE_WITH_INIT
#include "darknet_bridge.h"
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// #define PY_ARRAY_UNIQUE_SYMBOL shmem_array_api
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

%}

%init %{

//#ifndef GPU
//    gpu_index = -1;
//#else
//    if(gpu_index >= 0){
//        cuda_set_device(gpu_index);
//    }
//#endif

import_array(); // numpy initialization that should be run only once

%}

// Swig should not try to create a default constructor for the following classes as they're abstract (swig interface file should not have the constructors either):
// %nodefaultctor FrameFilter;
// %nodefaultctor Thread;

%typemap(in) (std::size_t) {
  $1=PyLong_AsSize_t($input);
}

%inline %{  
%}

// next, expose what is necessary
// autogenerate from this point on
 
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
 
class DarknetPredictor {                                                                     // <pyapi>
                                                                                             // <pyapi>
public:                                                                                      // <pyapi>
    DarknetPredictor(DarknetContext ctx, float thresh = .5, float hier_thresh = .5);         // <pyapi>
    virtual ~DarknetPredictor();                                                             // <pyapi>
public:                                                                   // <pyapi>
    PyObject* predict(PyObject* pyarr, bool draw = false);                // <pyapi>
    void setGpuIndex(int i);                                              // <pyapi>
};                                                                        // <pyapi>
 
class DarknetTrainer {                                                    // <pyapi>
public:                                                                   // <pyapi>
    DarknetTrainer(DarknetContext ctx, PyObject* py_gpu_list);            // <pyapi>
    virtual ~DarknetTrainer();                                            // <pyapi>
    void train();                                                         // <pyapi>
};                                                                        // <pyapi> 
