import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
import os
from shutil import copyfile
from time import sleep, strftime, localtime
from Loader import Loader
from configuration import config
from Utils import *
from layers import *
import librosa
import time


class FrontEnd(object):
    def __init__(self):
        self.reuse = False

    def run(self, input, print_shape=False, is_training=True):
        self.input = input
        self.is_training=is_training
        self.output = front_end_cnn("stft", self.input, reuse=self.reuse, print_shape=print_shape)
        self.reuse = True

        return self.output      


class Unet(object):
    def __init__(self, model):
        self.model = model
        self.reuse = False

    def run(self, input, is_training=True, print_shape=False):
        self.input = input

        if is_training == 1:
            self.is_training = True
        elif is_training == 0:
            self.is_training = False
        else:
            raise ValueError("Wrong value for in_training !!")

        input_shape = self.input.get_shape().as_list()
     
        if print_shape:
            print("\n-----------------------------------Seperator----------------------------------------")
            print("Separator input shape : {}".format(input_shape))

        with tf.variable_scope("UnetSeparator", reuse=self.reuse):
            """
            Using log
            """
            if config.is_log:
                self.in_r = self.input[:,:,:,0:1]
                self.in_i = self.input[:,:,:,1:2]
                self.in_mag = magnitude(self.input)
                self.in_phase = self.input/(self.in_mag + 1e-07)
                self.in_log_mag = tf.log(1e+2 * self.in_mag + 1.)
                if config.complex_mask:
                    self.input = self.in_log_mag * self.in_phase
                else:
                    self.input = self.in_log_mag
            else:
                pass

            if self.model == 'model10':
                # (512, 64)
                self.conv1 = conv2d_real("conv1", self.input, oc=45, f_h=7, f_w=5, s_h=2, s_w=2, bn=False, act='lrelu',
                                    print_shape=print_shape, is_training=self.is_training)
                # (256, 32)
                self.conv2 = conv2d_real("conv2", self.conv1, oc=90, f_h=7, f_w=5, s_h=2, s_w=2, bn=True, act='lrelu',
                                    print_shape=print_shape, is_training=self.is_training)
                # (128, 32)
                self.conv3 = conv2d_real("conv3", self.conv2, oc=90, f_h=5, f_w=3, s_h=2, s_w=2, bn=True, act='lrelu',
                                    print_shape=print_shape, is_training=self.is_training)
                # (64, 16)
                self.conv4 = conv2d_real("conv4", self.conv3, oc=90, f_h=5, f_w=3, s_h=2, s_w=2, bn=True, act='lrelu',
                                    print_shape=print_shape, is_training=self.is_training)                
                # (32, 16)
                self.conv5 = conv2d_real("conv5", self.conv4, oc=90, f_h=5, f_w=3, s_h=2, s_w=1, bn=True, act='lrelu',
                                    print_shape=print_shape, is_training=self.is_training)
                # (16, 8)
                self.deconv4 = deconv2d_real("deconv5", self.conv5, oc=90, f_h=5, f_w=3, s_h=2, s_w=1, bn=True, act='lrelu',
                                        print_shape=print_shape, is_training=self.is_training)
                self.d4 = tf.concat([self.deconv4, self.conv4], axis=-1)
                self.deconv3 = deconv2d_real("deconv4", self.d4, oc=90, f_h=5, f_w=3, s_h=2, s_w=2, bn=True, act='lrelu',
                                        print_shape=print_shape, is_training=self.is_training)
                self.d3 = tf.concat([self.deconv3, self.conv3], axis=-1)
                self.deconv2 = deconv2d_real("deconv3", self.d3, oc=90, f_h=5, f_w=3, s_h=2, s_w=2, bn=True, act='lrelu',
                                        print_shape=print_shape, is_training=self.is_training)
                self.d2 = tf.concat([self.deconv2, self.conv2], axis=-1)
                self.deconv1 = deconv2d_real("deconv2", self.d2, oc=45, f_h=7, f_w=5, s_h=2, s_w=2, bn=True, act='lrelu',
                                        print_shape=print_shape, is_training=self.is_training)                                                        
                self.d1 = tf.concat([self.deconv1, self.conv1], axis=-1)
                self.out = deconv2d_real("deconv1", self.d1, oc=2, f_h=7, f_w=5, s_h=2, s_w=2, bn=False, act=None,
                                        print_shape=print_shape, is_training=self.is_training)

            elif self.model == 'model20':
                # (512, 64)
                self.conv1 = conv2d_real("conv1", self.input, oc=45, f_h=7, f_w=1, s_h=1, s_w=1, bn=False, act='lrelu',
                                    print_shape=print_shape, is_training=self.is_training)
                # (512, 64)
                self.conv2 = conv2d_real("conv2", self.conv1, oc=45, f_h=1, f_w=7, s_h=1, s_w=1, bn=True, act='lrelu',
                                    print_shape=print_shape, is_training=self.is_training)
                # (512, 64)
                self.conv3 = conv2d_real("conv3", self.conv2, oc=90, f_h=7, f_w=5, s_h=2, s_w=2, bn=True, act='lrelu',
                                    print_shape=print_shape, is_training=self.is_training)
                # (256, 32)
                self.conv4 = conv2d_real("conv4", self.conv3, oc=90, f_h=7, f_w=5, s_h=2, s_w=2, bn=True, act='lrelu',
                                    print_shape=print_shape, is_training=self.is_training)                
                # (128, 32)
                self.conv5 = conv2d_real("conv5", self.conv4, oc=90, f_h=5, f_w=3, s_h=2, s_w=2, bn=True, act='lrelu',
                                    print_shape=print_shape, is_training=self.is_training)
                # (64, 16)
                self.conv6 = conv2d_real("conv6", self.conv5, oc=90, f_h=5, f_w=3, s_h=2, s_w=2, bn=True, act='lrelu',
                                    print_shape=print_shape, is_training=self.is_training)
                # (64, 16)
                self.conv7 = conv2d_real("conv7", self.conv6, oc=90, f_h=5, f_w=3, s_h=2, s_w=1, bn=True, act='lrelu',
                                    print_shape=print_shape, is_training=self.is_training)
                # (8, 4)
                self.conv8 = conv2d_real("conv8", self.conv7, oc=180, f_h=5, f_w=3, s_h=2, s_w=1, bn=True, act='lrelu',
                                    print_shape=print_shape, is_training=self.is_training)

                self.deconv7 = deconv2d_real("deconv8", self.conv8, oc=90, f_h=5, f_w=3, s_h=2, s_w=1, bn=True, act='lrelu',
                                    print_shape=print_shape, is_training=self.is_training)
                self.d7 = tf.concat([self.deconv7, self.conv7], axis=-1)
                self.deconv6 = deconv2d_real("deconv7", self.d7, oc=90, f_h=5, f_w=3, s_h=2, s_w=1, bn=True, act='lrelu',
                                        print_shape=print_shape, is_training=self.is_training)
                self.d6 = tf.concat([self.deconv6, self.conv6], axis=-1)
                self.deconv5 = deconv2d_real("deconv6", self.d6, oc=90, f_h=5, f_w=3, s_h=2, s_w=2, bn=True, act='lrelu',
                                        print_shape=print_shape, is_training=self.is_training)
                self.d5 = tf.concat([self.deconv5, self.conv5], axis=-1)
                self.deconv4 = deconv2d_real("deconv5", self.d5, oc=90, f_h=5, f_w=3, s_h=2, s_w=2, bn=True, act='lrelu',
                                        print_shape=print_shape, is_training=self.is_training)
                self.d4 = tf.concat([self.deconv4, self.conv4], axis=-1)
                self.deconv3 = deconv2d_real("deconv4", self.d4, oc=90, f_h=7, f_w=5, s_h=2, s_w=2, bn=True, act='lrelu',
                                        print_shape=print_shape, is_training=self.is_training)
                self.d3 = tf.concat([self.deconv3, self.conv3], axis=-1)
                self.deconv2 = deconv2d_real("deconv3", self.d3, oc=45, f_h=7, f_w=5, s_h=2, s_w=2, bn=True, act='lrelu',
                                        print_shape=print_shape, is_training=self.is_training)
                self.d2 = tf.concat([self.deconv2, self.conv2], axis=-1)
                self.deconv1 = deconv2d_real("deconv2", self.d2, oc=45, f_h=1, f_w=7, s_h=1, s_w=1, bn=True, act='lrelu',
                                        print_shape=print_shape, is_training=self.is_training)                                                        
                self.d1 = tf.concat([self.deconv1, self.conv1], axis=-1)
                self.out = deconv2d_real("deconv1", self.d1, oc=2, f_h=7, f_w=1, s_h=1, s_w=1, bn=False, act=None,
                                        print_shape=print_shape, is_training=self.is_training)
            else :
                raise AssertionError("Wrong model configuration !!")

            if config.complex_mask:
                self.mask = tanh_squash(self.out, reuse=self.reuse)
            else:
                self.mask = tf.nn.sigmoid(self.out)
                self.mask = tf.concat([self.mask[:,:,:,0:1], tf.zeros_like(self.mask[:, :, :, 0:1])], axis=-1)

        self.reuse = True
        return self.mask


class BiLSTM(object):
    def __init__(self, model):
        self.model = model
        self.reuse = False

    def run(self, input, is_training=True, print_shape=False):
        self.input = input

        if is_training == 1:
            self.is_training = True
        elif is_training == 0:
            self.is_training = False

        input_shape = self.input.get_shape().as_list()

        if print_shape:
            print("\n-----------------------------------Seperator----------------------------------------")
            print("Separator input shape : {}".format(input_shape))

        with tf.variable_scope("BiLSTMSeparator", reuse=self.reuse):
            """
            Using log
            """
            if config.is_log:
                self.in_r = self.input[:, :, :, 0:1]
                self.in_i = self.input[:, :, :, 1:2]
                self.in_mag = magnitude(self.input)
                self.in_phase = self.input / (self.in_mag + 1e-07)
                self.in_log_mag = tf.log(1e+2 * self.in_mag + 1.)
                self.input = self.in_log_mag * self.in_phase
            else:
                pass

            self.input = tf.transpose(self.input, perm=[0, 2, 1, 3])
            self.input = tf.concat([self.input[:, :, :, 0], self.input[:, :, :, 1]], axis=-1)

            if self.model == 'model10':
                cells_fw = [tf.contrib.rnn.LSTMCell(num_units=224) for i in range(1)]
                cells_bw = [tf.contrib.rnn.LSTMCell(num_units=224) for i in range(1)]
                self.lstm, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw, cells_bw=cells_bw, inputs=self.input,
                                                                             dtype=tf.float32, time_major=False)
                self.out = tf.contrib.layers.fully_connected(self.lstm, config.nfft, activation_fn=None)

            elif self.model == 'model20':
                cells_fw = [tf.contrib.rnn.LSTMCell(num_units=224) for i in range(2)]
                cells_bw = [tf.contrib.rnn.LSTMCell(num_units=224) for i in range(2)]
                self.lstm, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw, cells_bw=cells_bw, inputs=self.input,
                                                                             dtype=tf.float32, time_major=False)
                self.out = tf.contrib.layers.fully_connected(self.lstm, config.nfft, activation_fn=None)

            else :
                raise AssertionError("Wrong model configuration !!")

            self.out = tf.stack([self.out[:, :, :config.nfft // 2], self.out[:, :, config.nfft // 2:]], axis=-1)
            self.out = tf.transpose(self.out, perm=[0, 2, 1, 3])
            self.mask = tanh_squash(self.out, reuse=self.reuse)

        self.reuse = True
        return self.mask


class LSTM(object):
    def __init__(self, model):
        self.model = model
        self.reuse = False

    def run(self, input, is_training=True, print_shape=False):
        self.input = input

        if is_training == 1:
            self.is_training = True
        elif is_training == 0:
            self.is_training = False

        input_shape = self.input.get_shape().as_list()

        if print_shape:
            print("\n-----------------------------------Seperator----------------------------------------")
            print("Separator input shape : {}".format(input_shape))

        with tf.variable_scope("LSTMSeparator", reuse=self.reuse):
            """
            Using log
            """
            if config.is_log:
                self.in_r = self.input[:, :, :, 0:1]
                self.in_i = self.input[:, :, :, 1:2]
                self.in_mag = magnitude(self.input)
                self.in_phase = self.input / (self.in_mag + 1e-07)
                self.in_log_mag = tf.log(1e+2 * self.in_mag + 1.)
                self.input = self.in_log_mag * self.in_phase
            else:
                pass

            self.input = tf.transpose(self.input, perm=[0, 2, 1, 3])
            self.input = tf.concat([self.input[:, :, :, 0], self.input[:, :, :, 1]], axis=-1)

            if self.model == 'model10':
                lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=256) for i in range(2)]
                lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)
                self.lstm, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=self.input, dtype=tf.float32, time_major=False)

                self.out = tf.contrib.layers.fully_connected(self.lstm, config.nfft, activation_fn=None)

            elif self.model == 'model20':
                lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=320) for i in range(3)]
                lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)
                self.lstm, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=self.input, dtype=tf.float32, time_major=False)

                self.out = tf.contrib.layers.fully_connected(self.lstm, config.nfft, activation_fn=None)
            else :
                raise AssertionError("Wrong model configuration !!")

            self.out = tf.stack([self.out[:, :, :config.nfft // 2], self.out[:, :, config.nfft // 2:]], axis=-1)
            self.out = tf.transpose(self.out, perm=[0, 2, 1, 3])
            self.mask = tanh_squash(self.out, reuse=self.reuse)

        self.reuse = True
        return self.mask


class Dilated(object):
    def __init__(self, model):
        self.model = model
        self.reuse = False

    def run(self, input, is_training=True, print_shape=True):
        self.input = input

        if is_training == 1:
            self.is_training = True
        elif is_training == 0:
            self.is_training = False

        input_shape = self.input.get_shape().as_list()

        if print_shape:
            print("\n-----------------------------------Seperator----------------------------------------")
            print("Separator input shape : {}".format(input_shape))

        with tf.variable_scope("DilatedSeparator", reuse=self.reuse):
            wav = tf.expand_dims(input, axis=-1)

            if self.model == 'model10':
                wav1 = conv1d("conv0", wav, oc=128, f_w=20, s_w=10, d_w=1, bn=False, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav2 = conv1d("conv1", wav1, oc=128, f_w=3, s_w=1, d_w=1, bn=False, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav3 = conv1d("conv2", wav2, oc=256, f_w=3, s_w=1, d_w=2, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav4 = conv1d("conv3", wav3, oc=256, f_w=3, s_w=1, d_w=4, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav5 = conv1d("conv4", wav4, oc=512, f_w=3, s_w=1, d_w=8, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav6 = conv1d("conv5", wav5, oc=256, f_w=3, s_w=1, d_w=16, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav7 = conv1d("conv6", wav6, oc=128, f_w=3, s_w=1, d_w=32, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav8 = conv1d("conv7", wav7, oc=128, f_w=3, s_w=1, d_w=64, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav9 = conv1d("conv8", wav8, oc=128, f_w=3, s_w=1, d_w=128, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav10 = conv1d("conv9", wav9, oc=128, f_w=3, s_w=1, d_w=256, bn=False, is_training=is_training,
                             print_shape=print_shape,
                             act='sigmoid')

                wav10 = wav10 * wav1
                self.out = deconv1d("conv10", wav10, oc=1, f_w=20, s_w=10, bn=False, is_training=is_training,
                               print_shape=print_shape,
                               act='none')

            elif self.model == 'model20':
                wav1 = conv1d("conv0", wav, oc=128, f_w=20, s_w=10, d_w=1, bn=False, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav2 = conv1d("conv1", wav1, oc=128, f_w=3, s_w=1, d_w=1, bn=False, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav3 = conv1d("conv2", wav2, oc=256, f_w=3, s_w=1, d_w=2, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav4 = conv1d("conv3", wav3, oc=256, f_w=3, s_w=1, d_w=4, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav5 = conv1d("conv4", wav4, oc=512, f_w=3, s_w=1, d_w=8, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav6 = conv1d("conv5", wav5, oc=256, f_w=3, s_w=1, d_w=16, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav7 = conv1d("conv6", wav6, oc=128, f_w=3, s_w=1, d_w=32, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav8 = conv1d("conv7", wav7, oc=128, f_w=3, s_w=1, d_w=64, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav9 = conv1d("conv8", wav8, oc=128, f_w=3, s_w=1, d_w=128, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav10 = conv1d("conv9", wav9, oc=128, f_w=3, s_w=1, d_w=256, bn=False, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')

                # wav10 = tf.concat([wav10, wav1], axis=-1)

                wav11 = conv1d("conv2-1", wav10, oc=128, f_w=3, s_w=1, d_w=1, bn=False, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav12 = conv1d("conv2-2", wav11, oc=256, f_w=3, s_w=1, d_w=2, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav13 = conv1d("conv2-3", wav12, oc=256, f_w=3, s_w=1, d_w=4, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav14 = conv1d("conv2-4", wav13, oc=512, f_w=3, s_w=1, d_w=8, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav15 = conv1d("conv2-5", wav14, oc=256, f_w=3, s_w=1, d_w=16, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav16 = conv1d("conv2-6", wav15, oc=128, f_w=3, s_w=1, d_w=32, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav17 = conv1d("conv2-7", wav16, oc=128, f_w=3, s_w=1, d_w=64, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav18 = conv1d("conv2-8", wav17, oc=128, f_w=3, s_w=1, d_w=128, bn=True, is_training=is_training,
                             print_shape=print_shape,
                             act='lrelu')
                wav19 = conv1d("conv2-9", wav18, oc=128, f_w=3, s_w=1, d_w=256, bn=False, is_training=is_training,
                             print_shape=print_shape,
                             act='sigmoid')

                wav19 = wav19 * wav1
                self.out = deconv1d("conv10", wav19, oc=1, f_w=20, s_w=10, bn=False, is_training=is_training,
                               print_shape=print_shape,
                               act='none')
            else:
                raise AssertionError("Wrong model configuration !!")

        self.reuse = True
        return self.out[:,:,0]


class BackEnd(object):
    def __init__(self):
        self.reuse = False

    def run(self, input, is_training=True, print_shape=False):
        self.input = input
        self.is_training=is_training
        self.output = back_end_cnn(name='istft', input=self.input, N=config.nfft, stride=config.stride, reuse=self.reuse, print_shape=print_shape)
        self.reuse = True
        return self.output
