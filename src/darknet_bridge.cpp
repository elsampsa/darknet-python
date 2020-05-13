/*
 * darknet_bridge.cpp : Darket cpp and python interfaces
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
 *  @file    darknet_bridge.cpp
 *  @author  Sampsa Riikonen
 *  @date    2018
 *  @version 0.2.0 
 *  
 *  @brief   Darket cpp and python interfaces
 */ 

#include "darknet_bridge.h"

// #define SKIP_DARKNET 1
// #define DEBUG 1


bool darknet_with_cuda() {
    #ifdef GPU
    return true;
    #else
    return false;
    #endif
}

image **load_alphabet_custom(const char* path)
{
    int i, j;
    const int nsize = 8;
    image **alphabets = (image**)calloc(nsize, sizeof(image*)); // should be size of the pointer, not of the image, right?
    for(j = 0; j < nsize; ++j){
        alphabets[j] = (image*)calloc(128, sizeof(image));
        for(i = 32; i < 127; ++i){
            char buff[256];
            // sprintf(buff, "data/labels/%d_%d.png", i, j);
            sprintf(buff, "%s/%d_%d.png", path, i, j);
            // std::cout << "load_alphabet_custom : " << buff << " : " << path << std::endl;
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}


list *read_data_cfg_custom(std::string datacfg) { // let's read the config file from a string instead
    // FILE *fmemopen(void *buf, size_t size, const char *mode); 
    // FILE *file = fopen(filename, "r");
    FILE *file = fmemopen((void*)datacfg.c_str(), datacfg.length(), "r");
    
    // if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;   
}


PyObject *get_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes)
// draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
{
    PyObject *pylist = PyList_New(0);
    int i, j;

    // std::cout << "get_detections: num, classes : " << num << " " << classes << std::endl;
    
    for(i = 0; i < num; ++i){ // loop over boxes (i.e. detections)
        // char labelstr[4096] = {0};
        // int class = -1;
        for(j = 0; j < classes; ++j){
            // printf("> %s: %.0f%%\n", names[j], dets[i].prob[j]*100);
            if (dets[i].prob[j] > thresh){
                /*
                if (class < 0) {
                    strcat(labelstr, names[j]);
                    class = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                */
                box b = dets[i].bbox;
                //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

                int left  = (b.x-b.w/2.)*im.w;
                int right = (b.x+b.w/2.)*im.w;
                
                // darknet coordinate system starts from upper left corner (data coordinates)
                // we want from the lower left corner (visual coordinates)
                // so, we use 1.-
                int top   = (1.-(b.y+b.h/2.))*im.h;
                int bot   = (1.-(b.y-b.h/2.))*im.h;

                if(left < 0) left = 0;
                if(right > im.w-1) right = im.w-1;
                if(top < 0) top = 0;
                if(bot > im.h-1) bot = im.h-1;
                
                // printf("%s: (%i, %i, %i, %i) %.0f%%\n", names[j], left, right, top, bot, dets[i].prob[j]*100);
                
                PyObject *pytuple = PyTuple_Pack(6, // class name, probability in %, left, right, top, bottom
                    PyUnicode_FromString(names[j]),
                    PyLong_FromLong((long)(dets[i].prob[j]*100)),
                    PyLong_FromLong((long)left),
                    PyLong_FromLong((long)right),
                    PyLong_FromLong((long)top),
                    PyLong_FromLong((long)bot)
                );
                int success = PyList_Append(pylist, pytuple);
            }
        }
    }
    
    return pylist;
}


// this is not included in libdarknet.a !  .. so we must repeat it here.  It's from file "detector.c"
void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg_custom(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network **nets = (network**)calloc(ngpus, sizeof(network)); // fixed cast

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 64;

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}



DarknetPredictor::DarknetPredictor(DarknetContext ctx, float thresh, float hier_thresh) : ctx(ctx), thresh(thresh), hier_thresh(hier_thresh), height(0), width(0), channels(0), my_gpu_index(-1) {
    // stupid const char => char conversions
    char* datacfg = strdup(ctx.datacfg.c_str());
    char* cfgfile = strdup(ctx.cfgfile.c_str()); 
    char* weightfile = strdup(ctx.weightfile.c_str());
    
    // list *options = read_data_cfg(datacfg);
    list *options = read_data_cfg_custom(ctx.datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    
#ifdef DEBUG
    std::cout << "DarknetPredictor: 1" << std::endl;
#endif
    
    names = get_labels(name_list);
    
#ifdef DEBUG
    std::cout << "DarknetPredictor: 2" << std::endl;
#endif
    
#ifdef DEBUG
    std::cout << "DarknetPredictor: datadir : " << ctx.datadir << " : " << ctx.datadir.c_str() << std::endl;
#endif
    alphabet = load_alphabet_custom(ctx.datadir.c_str());
    
#ifdef DEBUG    
    std::cout << "DarknetPredictor : " << cfgfile << " " << weightfile << std::endl;
#endif
    
#ifdef SKIP_DARKNET
#else
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    #ifdef DEBUG
    std::cout << "DarknetPredictor : net w, h=" << net->w << " " << net->h << std::endl;
    #endif
#endif
    srand(2222222);
    
    free(datacfg);
    free(cfgfile);
    free(weightfile);
    free_list(options);
    // free(name_list);
    
    im.data = NULL;
    im.w = 0;
    im.h = 0;
    im.c = 0;
}


DarknetPredictor::~DarknetPredictor() {
    
#ifdef SKIP_DARKNET
#else
    std::cout << "DarknetPredictor: releasing network" << std::endl;
    free_network(net); 
    // that doesn't free all gpu memory
    // pjreddie's darknet is a memory-leaking piece of shit
    // should never try to adapt code made by academics to production use
#endif
    // TODO:
    //free(alphabet); // or should run through the alphabets ..?
    //free_ptrs((void**)alphabet, 128);
}


void DarknetPredictor::pyArrayToImage(PyArrayObject* pyarr) {
    // input: 8-bit image data
    // if last dimensions is three, then this is just RGB24
    // the darknet image structure looks like this:
    /*
        typedef struct {
        int w;
        int h;
        int c;
        float *data;
    } image;
    */

    if (PyArray_NDIM(pyarr)<3) { // check number of dimensions : width, height, channels required
        im.c = 0; // c=0 indicates no success
    }
    
    npy_intp *dims = PyArray_DIMS(pyarr); // the dimensions
    
    // in python, the most rapid-running index (here the color) is the last
    int h = dims[0];
    int w = dims[1];
    int c = dims[2];
    
    if (h!=height or w!=width or c!=channels) { // its time to reallocate
        if (im.data) { // e.g. not NULL
            free_image(im);
        }
        im = make_image(w, h, c); // reserve im.data
    }
    unsigned char *data = (unsigned char *)PyArray_BYTES(pyarr);
    // int step = src->widthStep;
    int step = w*c;
    
    // transform from RGB24 and similar to darknet's internal format
    int i, j, k;
    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
}


PyObject* DarknetPredictor::predict(PyObject* pyarr, bool draw) {
    PyObject* feature_list;
    
    pyArrayToImage((PyArrayObject*)pyarr); // manipulates the member "im"
    if (im.c == 0) {
        return PyList_New(0);
    }
    
    // image im = load_image_color(input,0,0); // RGB24 image
    /*
    image.c : load_image_color 
        => load_image
            =>                
            #ifdef OPENCV
                image out = load_image_cv(filename, c);
                =>
                    image_opencv.cpp : load_image_cv
                        => image im = mat_to_image(m)
                            => rgbgr_image(im)  // switches between rgb and bgr .. so darknet uses rgb
            #else
                image out = load_image_stb(filename, c);
            #endif
    */
    
#ifdef SKIP_DARKNET
    return feature_list;
#endif
    
#ifdef DEBUG
    std::cout << "predict: net->w, net->h : " << net->w << " " << net->h << std::endl;
#endif
    
    image sized = letterbox_image(im, net->w, net->h); // TODO: avoid constant reallocations
    layer l = net->layers[net->n-1];
    
    // gpu_index : defined as extern int in darknet.h
    // .. used in detector.c
    // cuda.c states "int gpu_index = 0"
    // so it's a singleton living in cuda.c
    // network structure has als a gpu_index member (see darknet.h)
    // network->gpu_index
    
    #ifdef GPU
    // which one I should use ..?
    if (my_gpu_index>-1) {
        cuda_set_device(my_gpu_index);
        net->gpu_index = my_gpu_index;
    }
    #endif
    
    float *X = sized.data;
    // time=what_time_is_it_now();
    network_predict(net, X);
    // printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
    
    int nboxes = 0;
    detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
    
    //printf("%d\n", nboxes);
    //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    
    float nms = .45;
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
    
    if (draw) {
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        // draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes, 0); // sowson_darknet
    }

    // std::cout << "DarknetPredictor: predict: " << std::endl;
    feature_list = get_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
    
    // TODO: avoid constant reallocations
    free_detections(dets, nboxes);
    free_image(sized); 
    
    return feature_list;
}

void DarknetPredictor::setGpuIndex(int i) {
    my_gpu_index=i;
}



DarknetTrainer::DarknetTrainer(DarknetContext ctx, PyObject* py_gpu_list) : ctx(ctx) {
    /* // excerpt from the darknet c code:
    
    // input pars
    int clear = find_arg(argc, argv, "-clear");
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    
    // aux
    int gpu = 0;
    
    // class members
    int *gpus = 0;
    int ngpus = 0;
    
    // gpu_index is a singleton
    */
    
    /*
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }
    */
    
    clear = 0; // ?
    
    gpus = &gpu_index;
    ngpus = 1;
    
    // so
    
    // TODO: get the gpu indexes from the python list
    
}
 
DarknetTrainer::~DarknetTrainer() {
}
 
 
void DarknetTrainer::train() {
    // stupid const char => char conversions
    char* datacfg = strdup(ctx.datacfg.c_str());
    char* cfgfile = strdup(ctx.cfgfile.c_str()); 
    char* weightfile = strdup(ctx.weightfile.c_str());
    
    // simply ..
    train_detector(datacfg, cfgfile, weightfile, gpus, ngpus, clear);
}

