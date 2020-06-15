import tensorflow as tf
from glob import glob
import os
from shutil import copyfile
from time import sleep, strftime, localtime
from Loader import Loader
from configuration import get_config
from Utils import *
from layers import *
import librosa
import time
import numpy as np

config = get_config()


class build_graph(object):    
    def __init__(self, train=True):
        self.train=train

    def __call__(self, input, functions):
        self.mix = input["mix"]
        input_shape = self.mix.get_shape().as_list()
        self.alpha = tf.constant(0.5, shape=[input_shape[0],1])

        frontend = functions[0]
        separator = functions[1]
        backend = functions[2]

        if self.train:
            state = 'train'
            self.is_training = np.array([1.], dtype=np.float32)
        else:
            state = 'test'
            self.is_training = np.array([0.], dtype=np.float32)

        self.source = input["source"]
        self.input_spec = frontend.run(self.mix, print_shape=self.train)
        self.input_mag = magnitude(self.input_spec)
        self.source_spec = frontend.run(self.source, print_shape=False)

        if config.hybrid:
            print("\nHybrid Network !!")
            separator2 = functions[3]

            self.mask1_mid = separator.run(self.input_spec, is_training=self.is_training, print_shape=self.train)
            self.masked_spec_r1_mid = self.input_spec[:, :, :, 0:1] * self.mask1_mid[:, :, :, 0:1] - \
                                  self.input_spec[:, :, :, 1:2] * self.mask1_mid[:, :, :, 1:2]
            self.masked_spec_i1_mid = self.input_spec[:, :, :, 0:1] * self.mask1_mid[:, :, :, 1:2] + \
                                  self.input_spec[:, :, :, 1:2] * self.mask1_mid[:, :, :, 0:1]
            self.masked_spec1_mid = tf.concat((self.masked_spec_r1_mid, self.masked_spec_i1_mid), axis=-1)
            self.estimated1_mid = backend.run(self.masked_spec1_mid, print_shape=self.train)

            self.estimated1 = separator2.run(self.estimated1_mid, is_training=self.is_training, print_shape=self.train)
            self.masked_spec1 = frontend.run(self.estimated1, print_shape=self.train)

            self.estimated2_mid = separator2.run(self.mix, is_training=self.is_training, print_shape=self.train)
            self.masked_spec2_mid = frontend.run(self.estimated2_mid, print_shape=self.train)

            self.mask2 = separator.run(self.masked_spec2_mid, is_training=self.is_training, print_shape=self.train)
            self.masked_spec_r2 = self.masked_spec2_mid[:, :, :, 0:1] * self.mask2[:, :, :, 0:1] - \
                                  self.masked_spec2_mid[:, :, :, 1:2] * self.mask2[:, :, :, 1:2]
            self.masked_spec_i2 = self.masked_spec2_mid[:, :, :, 0:1] * self.mask2[:, :, :, 1:2] + \
                                  self.masked_spec2_mid[:, :, :, 1:2] * self.mask2[:, :, :, 0:1]
            self.masked_spec2 = tf.concat((self.masked_spec_r2, self.masked_spec_i2), axis=-1)
            self.estimated2 = backend.run(self.masked_spec2, print_shape=self.train)

            tf.identity(self.estimated1_mid, name='estimation_{}1_mid'.format(state))
            tf.identity(self.masked_spec1_mid, name='masked_spec_{}1_mid'.format(state))
            tf.identity(self.estimated2_mid, name='estimation_{}2_mid'.format(state))
            tf.identity(self.masked_spec2_mid, name='masked_spec_{}2_mid'.format(state))

            self.estimated = 0.5 * self.estimated1 + 0.5 * self.estimated2
            self.masked_spec = frontend.run(self.estimated, print_shape=False)

        else:
            if config.network=="dilated":
                self.estimated1 = separator.run(self.mix, is_training=self.is_training, print_shape=self.train)
                self.masked_spec1 = frontend.run(self.estimated1, print_shape=self.train)
            else:
                self.mask1 = separator.run(self.input_spec, is_training=self.is_training, print_shape=self.train)
                self.masked_spec_r1 = self.input_spec[:,:,:,0:1]*self.mask1[:,:,:,0:1] -\
                                     self.input_spec[:,:,:,1:2]*self.mask1[:,:,:,1:2]
                self.masked_spec_i1 = self.input_spec[:,:,:,0:1]*self.mask1[:,:,:,1:2] +\
                                     self.input_spec[:,:,:,1:2]*self.mask1[:,:,:,0:1]
                self.masked_spec1 = tf.concat((self.masked_spec_r1,self.masked_spec_i1),axis=-1)
                self.estimated1 = backend.run(self.masked_spec1, print_shape=self.train)

            if len(functions) >= 4:
                separator2 = functions[3]
                confidence = functions[4]
                if state == 'train':
                    print("\nEesemble model !!")

                if config.network2 == "dilated":
                    self.estimated2 = separator2.run(self.mix, is_training=self.is_training, print_shape=self.train)
                    self.masked_spec2 = frontend.run(self.estimated2, print_shape=False)
                else:
                    self.mask2 = separator2.run(self.input_spec, is_training=self.is_training, print_shape=self.train)
                    self.masked_spec_r2 = self.input_spec[:, :, :, 0:1] * self.mask2[:, :, :, 0:1] - \
                                         self.input_spec[:, :, :, 1:2] * self.mask2[:, :, :, 1:2]
                    self.masked_spec_i2 = self.input_spec[:, :, :, 0:1] * self.mask2[:, :, :, 1:2] + \
                                         self.input_spec[:, :, :, 1:2] * self.mask2[:, :, :, 0:1]
                    self.masked_spec2 = tf.concat((self.masked_spec_r2, self.masked_spec_i2), axis=-1)
                    self.estimated2 = backend.run(self.masked_spec2, print_shape=False)

                if config.confidence == "wave":
                    if config.confidence_input:
                        self.alpha = confidence.run(tf.stack([self.mix, self.estimated1, self.estimated2], axis=-1))
                    else:
                        self.alpha = confidence.run(tf.expand_dims(self.mix, axis=-1))
                elif config.confidence == "spec":
                    self.alpha = confidence.run(self.input_spec)

                self.estimated = self.alpha * self.estimated1 + (1 - self.alpha) * self.estimated2
                self.masked_spec = frontend.run(self.estimated, print_shape=False)

            else :
                self.estimated = self.estimated1
                self.masked_spec = self.masked_spec1
                self.estimated2 = self.estimated1
                self.masked_spec2 = self.masked_spec1

        tf.identity(self.alpha, name='alpha_{}'.format(state))
        tf.identity(self.input_spec, name='input_spec_{}'.format(state))
        tf.identity(self.source_spec, name='source_spec_{}'.format(state))

        tf.identity(self.estimated, name='estimation_{}'.format(state))
        tf.identity(self.masked_spec, name='masked_spec_{}'.format(state))
        tf.identity(self.estimated1, name='estimation_{}1'.format(state))
        tf.identity(self.masked_spec1, name='masked_spec_{}1'.format(state))
        tf.identity(self.estimated2, name='estimation_{}2'.format(state))
        tf.identity(self.masked_spec2, name='masked_spec_{}2'.format(state))

