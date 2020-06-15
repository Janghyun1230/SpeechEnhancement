# -*- coding: utf-8 -*-
from __future__ import print_function
from tensorflow.python.ops import init_ops
import tensorflow as tf
import numpy as np
import math as m
import librosa
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from configuration import config
from numpy.random import RandomState
from scipy import signal
from glob import glob
from tensorflow.python.layers import utils


def magnitude(input, name='mag'):
    input_r = input[:,:,:,0:1]
    input_i = input[:,:,:,1:2]
    input_mag = tf.sqrt(tf.square(input_r)+tf.square(input_i)+1e-13)
    tf.identity(input_mag, name=name)
    return input_mag


def real_imag(input):
    c= int(int(input.get_shape()[-1])/2)
    return input[:,:,:,:c], input[:,:,:,c:]


def leaky_relu(input, dtype=tf.float32, reuse=False):
    with tf.variable_scope("leaky_relu", reuse=reuse):
        b = tf.get_variable('radius', initializer=0.5, dtype= dtype)
        x, y = real_imag(input)
        mag = tf.sqrt(x ** 2 + y ** 2 + 1e-07)
        mask = tf.cast(tf.greater(mag, b), dtype=dtype)*0.8 + 0.2
        x = mask * x
        y = mask * y
        output= tf.concat([x,y], axis= -1)
    return output


def tanh_squash(input, reuse):
    x, y = real_imag(input)
    mag = tf.sqrt(x ** 2 + y ** 2 + 1e-12)
    mag_mask = tf.nn.tanh(mag)
    phase_mask = input/(mag+1e-13)
    output = mag_mask * phase_mask

    return output


def front_end_cnn(name, 
                  input,
                  N=config.nfft, 
                  stride=config.stride, 
                  reuse=False, 
                  padding="SAME", 
                  dtype= tf.float32,
                  print_shape= False,
                  dc_trainable= config.dc_trainable,
                  fft_trainable= config.fft_trainable,
                  hanning= config.hanning,
                  is_multi= config.is_multi
                  ):
    """
    Front end convolutional layer

    Args:
        name: name of the convnet layer
        input: input of the convnet layer
        N: "dft size" or "output channel"
        stride: "hop size" or "stride size"
        reuse: reuse layer(True) or not(False)

    Return:
        The output of the front end convnet layer

    Raises:
        Nothing

    """
    if print_shape:
        print("\n-----------------------------------Front end----------------------------------------")
        print("input shape : ", input.get_shape()) # (16,16384)

    if not is_multi:
        input = tf.expand_dims(input,axis=2)

    I = np.eye(N)
    W = np.fft.fft(I) # DFT matrix
    A = np.real(W[:,:int(N/2)]) # (filter_size=N, input_channel=1, output_channel=int(N/2))
    B = np.imag(W[:,:int(N/2)]) # (filter_size=N, input_channel=1, output_channel=int(N/2))

    A = A.astype("float32")
    B = B.astype("float32")

    if hanning == True:
        A*= np.expand_dims(signal.get_window('hanning', N), axis=1)
        B*= np.expand_dims(signal.get_window('hanning', N), axis=1)
        if print_shape== True:
            print("use hanning window !")
        
    R_dc = A[:,0:1]/np.sqrt(N) #(2048, 1)
    R_rest = A[:,1:]/np.sqrt(N) #(2048, 1023)
    I_dc = B[:,0:1]/np.sqrt(N) #(2048, 1)
    I_rest = B[:,1:]/np.sqrt(N) #(2048, 1023)

    #[filter_width, in_channels, out_channels]

    R_dc = np.expand_dims(R_dc, axis=1) # (2048, 1) -> (2048,1,1)
    R_rest = np.expand_dims(R_rest, axis=1) # (2048, 1023) -> (2048,1,1023)

    I_dc = np.expand_dims(I_dc, axis=1) # (2048, 1) -> (2048,1,1)
    I_rest = np.expand_dims(I_rest, axis=1) # (2048, 1023) -> (2048,1,1023)

    if print_shape== True:
        print("real filter R_dc shape : ", R_dc.shape)
        print("real filter R_rest shape : ", R_rest.shape)

    with tf.variable_scope("FrontEnd"):    
        with tf.variable_scope(name, reuse=reuse):

            """ Initialize the conv filter as the Real part of Fourier Basis : A
                                    &
            Initialize the conv filter as the Imag part of Fourier Basis : B """

            FB_R_dc = tf.get_variable("RealFilter_dc", initializer=R_dc, dtype=dtype, trainable=dc_trainable) # (2048,1,1)
            FB_R_rest = tf.get_variable("RealFilter_rest", initializer=R_rest, dtype=dtype, trainable=fft_trainable) # (2048,1,1023)
            FB_R = tf.concat([FB_R_dc,FB_R_rest],axis=2) # (filter_size=2048, input_channel=1, output_channel=1024)

            FB_I_dc = tf.get_variable("ImagFilter_dc", initializer=I_dc, dtype=dtype, trainable=dc_trainable)
            FB_I_rest = tf.get_variable("ImagFilter_rest", initializer=I_rest, dtype=dtype, trainable=fft_trainable)
            FB_I = tf.concat([FB_I_dc,FB_I_rest],axis=2)

            if is_multi == False:
                R = tf.nn.conv1d(input, FB_R, stride=stride, padding=padding, data_format="NHWC")
                I = tf.nn.conv1d(input, FB_I, stride=stride, padding=padding, data_format="NHWC")

                # (16,44,1024) 1024*16/512 = 32
                R = tf.expand_dims(R, axis=1)
                I = tf.expand_dims(I, axis=1)
                output = tf.concat([R,I], axis=1)
            else:
                R_L = tf.nn.conv1d(input[:,:,0:1], FB_R, stride=stride, padding=padding, data_format="NHWC")
                R_R = tf.nn.conv1d(input[:,:,1:2], FB_R, stride=stride, padding=padding, data_format="NHWC")
                
                I_L = tf.nn.conv1d(input[:,:,0:1], FB_I, stride=stride, padding=padding, data_format="NHWC")
                I_R = tf.nn.conv1d(input[:,:,1:2], FB_I, stride=stride, padding=padding, data_format="NHWC")

                R_L = tf.expand_dims(R_L, axis=1)
                I_L = tf.expand_dims(I_L, axis=1)
                R_R = tf.expand_dims(R_R, axis=1)
                I_R = tf.expand_dims(I_R, axis=1)
                output = tf.concat([R_L, R_R, I_L, I_R], axis=1)
                
            output = tf.transpose(output, perm=[0,3,2,1])
            
    return output


def back_end_cnn(name,
                 input,
                 N=config.nfft,
                 stride=config.stride,
                 padding='SAME',
                 reuse=False,
                 audio_size=config.audio_size,
                 dtype= tf.float32,
                 print_shape= False,
                 ortho = config.ortho,
                 hanning= config.hanning,
                 is_multi= config.is_multi
                 ):
    """
    Back end convolutional layer

    Args:
        name: name of the convnet layer
        input: input of the convnet layer
        N: "dft size" or "output channel"
        stride: "hop size" or "stride size"
        reuse: reuse layer(True) or not(False)

    Return:
        The output of the front end convnet layer

    Raises:
        Nothing

    """
    input_shape = input.get_shape().as_list()

    if print_shape == True:
        print("\n-------------------------------------Back end---------------------------------------")

    # input: (16,1024,32,2) -> (16,1,32,2048)
    if is_multi == False:
        input_real = input[:,:,:,0:1]
        input_imag = input[:,:,:,1:2]
        input = tf.concat([input_real, input_imag], axis=1)
        input = tf.transpose(input, perm=[0,3,2,1])
    else :
        input_real = input[:,:,:,0:2]
        input_imag = input[:,:,:,2:4]
        input_L = tf.concat([input_real[:,:,:,0:1], input_imag[:,:,:,0:1]], axis=1)
        input_R = tf.concat([input_real[:,:,:,1:2], input_imag[:,:,:,1:2]], axis=1)
        input_L = tf.transpose(input_L, perm=[0,3,2,1])
        input_R = tf.transpose(input_R, perm=[0,3,2,1])
        
        
    """ Get Front End Filter and take Hermitian """
    if ortho == False:
        if print_shape == True:
            print("Not Orthogonal BackEnd !")
            print("input shape : ", input_shape) # [batch,1,32,2048]

        with tf.variable_scope("BackEnd"):
            with tf.variable_scope(name, reuse=reuse):
                I = np.eye(N)
                W = np.fft.fft(I) # DFT matrix
                A = np.real(W[:,:int(N/2)]) # (filter_size=N, input_channel=1, output_channel=int(N/2))
                B = np.imag(W[:,:int(N/2)]) # (filter_size=N, input_channel=1, output_channel=int(N/2))
                A = A.astype("float32")
                B = B.astype("float32")

                R_dc = A[:,0:1]/np.sqrt(N) #(2048, 1)
                R_rest = A[:,1:]/np.sqrt(N) #(2048, 1023)
                I_dc = B[:,0:1]/np.sqrt(N) #(2048, 1)
                I_rest = B[:,1:]/np.sqrt(N) #(2048, 1023)

                #[filter_width, in_channels, out_channels]

                R_dc = np.expand_dims(R_dc, axis=1) # (2048, 1) -> (2048,1,1)
                R_rest = np.expand_dims(R_rest, axis=1) # (2048, 1023) -> (2048,1,1023)

                I_dc = np.expand_dims(I_dc, axis=1) # (2048, 1) -> (2048,1,1)
                I_rest = np.expand_dims(I_rest, axis=1) # (2048, 1023) -> (2048,1,1023)            

                C_dc = tf.expand_dims(tf.get_variable("RealFilter_dc",initializer=R_dc, dtype=dtype,trainable=True), axis=0)
                C_rest = tf.expand_dims(tf.get_variable("RealFilter_rest",initializer=R_rest, dtype=dtype,trainable=True), axis=0)
                D_dc = -tf.expand_dims(tf.get_variable("ImagFilter_dc",initializer=I_dc, dtype=dtype,trainable=True), axis=0)
                D_rest = -tf.expand_dims(tf.get_variable("ImagFilter_rest",initializer=I_rest, dtype=dtype,trainable=True), axis=0)
    else:
        if print_shape == True:
            print("Orthogonal BackEnd !")
            print("input shape : ", input_shape) # [batch,1,32,2048]

        with tf.variable_scope("FrontEnd"):
            with tf.variable_scope("stft", reuse=True):
                # get_variable from front_end, make it untrainable, expand_dims, and take Hermitian(= -sign for Imag Filter)
                C_dc = tf.expand_dims(tf.get_variable("RealFilter_dc",trainable=False), axis=0)      #(1, 2048, 1, 1)
                C_rest = tf.expand_dims(tf.get_variable("RealFilter_rest",trainable=False), axis=0)  #(1, 2048, 1, 1023)
                D_dc = -tf.expand_dims(tf.get_variable("ImagFilter_dc",trainable=False), axis=0)     #(1, 2048, 1, 1)
                D_rest = -tf.expand_dims(tf.get_variable("ImagFilter_rest",trainable=False), axis=0) #(1, 2048, 1, 1023)

    with tf.variable_scope("BackEnd"):
        with tf.variable_scope(name):
            C = tf.concat([(C_dc)/2,C_rest],axis=3)   #(1, 2048, 1, 1024)
            D = tf.concat([(D_dc)/2,D_rest],axis=3)   #(1, 2048, 1, 1024)
        
            if print_shape== True:
                print("Back end RealFilter shape : ", C.get_shape())
                print("Back end ImagFilter shape", D.get_shape())
        
            F_R = tf.concat([C,-D],axis=3)  #(1, 2048, 1, 2048), Filter that is used to calculate the Real part of the output
            F_I = tf.concat([D,C],axis=3)   #(1, 2048, 1, 2048), Filter that is used to calculate the Imaginary part of the output
        
            output_shape = [input_shape[0], 1, audio_size, 1]    # [16, 1, audio_size, 1]

            if is_multi == False:
                R = tf.nn.conv2d_transpose(input, F_R, output_shape, strides=[1,1,stride,1], padding=padding, data_format="NHWC")
                I = tf.nn.conv2d_transpose(input, F_I, output_shape, strides=[1,1,stride,1], padding=padding, data_format="NHWC")
                R = tf.squeeze(2*R)
                I = tf.squeeze(2*I)

            else :
                R_L = tf.nn.conv2d_transpose(input_L, F_R, output_shape, strides=[1,1,stride,1], padding=padding, data_format="NHWC")
                I_L = tf.nn.conv2d_transpose(input_L, F_I, output_shape, strides=[1,1,stride,1], padding=padding, data_format="NHWC")
                R_L = tf.squeeze(2*R_L)
                I_L = tf.squeeze(2*I_L)
                
                R_R = tf.nn.conv2d_transpose(input_R, F_R, output_shape, strides=[1,1,stride,1], padding=padding, data_format="NHWC")
                I_R = tf.nn.conv2d_transpose(input_R, F_I, output_shape, strides=[1,1,stride,1], padding=padding, data_format="NHWC")
                R_R = tf.squeeze(2*R_R)
                I_R = tf.squeeze(2*I_R)
        
        # scaling output
        if hanning == False :
            if padding == 'VALID':
                scale_factor = np.zeros([audio_size], dtype=np.float32)
                for i in range(int(N/stride)):
                    scale_factor[(i*stride): (audio_size-i*stride)] += 1
                scale_factor = tf.constant(np.expand_dims(scale_factor, axis=0))

            elif padding == 'SAME':
                padding_size= int((m.ceil(audio_size/stride)-1)*stride + N - audio_size)
                scale_factor = np.zeros([(audio_size + padding_size)], dtype=np.float32)
                for i in range(int(N / stride)):
                    scale_factor[(i * stride): (audio_size + padding_size - i * stride)] += 1
                scale_factor = scale_factor[int(padding_size / 2): audio_size + int(padding_size / 2)]
                scale_factor = tf.constant(np.expand_dims(scale_factor, axis=0))
        else :
            if padding == 'VALID':
                scale_factor = np.zeros([audio_size], dtype=np.float32)
                hanning_square= np.square(signal.get_window('hanning', N))
                for i in range(int((audio_size-N) / stride)+1):
                    scale_factor[(i * N): ((i+1)*N)] += hanning_square
                scale_factor = tf.constant(np.expand_dims(scale_factor, axis=0))

            elif padding == 'SAME':
                padding_size= int((m.ceil(audio_size/stride)-1)*stride + N - audio_size)
                scale_factor = np.zeros([(audio_size + padding_size)], dtype=np.float32)
                hanning_square= np.square(signal.get_window('hanning', N))
                for i in range(int(audio_size / stride)):
                    scale_factor[(i * stride): (i*stride +N)] += hanning_square
                scale_factor = scale_factor[int(padding_size / 2): audio_size + int(padding_size / 2)]
                scale_factor = tf.constant(np.expand_dims(scale_factor, axis=0))
        
        if is_multi == False:        
            R = R / (scale_factor+1e-7) 
            return R
        else:
            R_L = R_L / (scale_factor+1e-7)
            R_R = R_R / (scale_factor+1e-7)
            return R_L, R_R


def load_sample(root=config.test_sample_root):
    """
    Load test mixtures samples
    """
    test_sample_path = glob(root)
    print(test_sample_path)
    audio_dict = {}
    for j,path in enumerate(test_sample_path):

        audio_slice_list = []

        print("Loading audio from : {}".format(path))
        y, sr = librosa.core.load(path, sr=config.sampling_rate, mono=True)
        num_frame = int(y.shape[0]/16384)

        for i in range(num_frame):
            start_point = int(i*16384)
            end_point = int(start_point + 16384)
            slice = y[start_point:end_point]

            audio_slice_list.append(slice)

        audio_dict['song{}_audio'.format(j)] = np.asarray(audio_slice_list)
        print("shape : ", audio_dict['song{}_audio'.format(j)].shape)
        audio_dict['song{}_slice_number'.format(j)] = len(audio_slice_list)
        print("song{} audio slice number : {}".format(j, audio_dict['song{}_slice_number'.format(j)]))

    for key, value in audio_dict.items() :
        print (key)

    return audio_dict
