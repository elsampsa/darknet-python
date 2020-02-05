#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "opencl.h"
#include "layer.h"

void forward_deconvolutional_layer_gpu(layer l, network net)
{
    int i;

    int m = l.size*l.size*l.n;
    int n = l.h*l.w;
    int k = l.c;

    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    for(i = 0; i < l.batch; ++i){
        cl_mem_ext a = l.weights_gpu;
        cl_mem_ext b = net.input_gpu; // + i*l.c*l.h*l.w;
        cl_mem_ext c = net.workspace_gpu;

        gemm_offset_gpu(0,1,m,n,k,1,a,0,m,b,i*l.c*l.h*l.w,n,0,c,0,n);

        col2im_gpu(net.workspace_gpu.mem, i*l.outputs, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output_gpu.mem);
    }
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    activate_array_gpu(l.output_gpu, l.batch*l.n*l.out_w*l.out_h, l.activation);
}

void backward_deconvolutional_layer_gpu(layer l, network net)
{
    int i;

    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    //if(net.delta_gpu) memset(net.delta_gpu, 0, l.batch*l.h*l.w*l.c*sizeof(float));

    for(i = 0; i < l.batch; ++i){
        int m = l.c;
        int n = l.size*l.size*l.n;
        int k = l.h*l.w;

        cl_mem_ext a = net.input_gpu; // + i*m*k;
        cl_mem_ext b = net.workspace_gpu;
        cl_mem_ext c = l.weight_updates_gpu;

        im2col_gpu(l.delta_gpu.add(l.delta_gpu, i*l.outputs, l.outputs).mem, 0, l.out_c, l.out_h, l.out_w,
                   l.size, l.stride, l.pad, b.mem);

        gemm_offset_gpu(1,0,m,n,k,1,a,i*m*k,k,b,0,k,1,c,0,n);

        if(net.delta_gpu.ptr){
            int m = l.c;
            int n = l.h*l.w;
            int k = l.size*l.size*l.n;

            cl_mem_ext a = l.weights_gpu;
            cl_mem_ext b = net.workspace_gpu;
            cl_mem_ext c = net.delta_gpu; // + i*n*m;

            gemm_offset_gpu(0,0,m,n,k,1,a,0,k,b,0,n,1,c,i*n*m,n);
        }
    }
}

void pull_deconvolutional_layer(layer l)
{
    opencl_pull_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
    opencl_pull_array(l.biases_gpu, l.biases, l.n);
    opencl_pull_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
    opencl_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        opencl_pull_array(l.scales_gpu, l.scales, l.n);
        opencl_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        opencl_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_deconvolutional_layer(layer l)
{
    opencl_push_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
    opencl_push_array(l.biases_gpu, l.biases, l.n);
    opencl_push_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
    opencl_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        opencl_push_array(l.scales_gpu, l.scales, l.n);
        opencl_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        opencl_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void update_deconvolutional_layer_gpu(layer l, update_args a)
{
    int batch = a.batch;
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;

    int size = l.size*l.size*l.c*l.n;
    axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
    scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

    if(l.scales_gpu.ptr){
        axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
        scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
    }

    axpy_gpu(size, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
    axpy_gpu(size, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
    scal_gpu(size, momentum, l.weight_updates_gpu, 1);
}

