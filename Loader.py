from glob import glob
from os import walk, mkdir
from tfrecordreadwrite import read_and_decode
import math, os
import tensorflow as tf
import numpy as np
from configuration import config


class Loader(object):
    def __init__(self, root, batch_size,
                 split=None, seed=None,
                 audio_size= config.audio_size):

        self.root = root
        self.batch_size = batch_size
        self.split = split
        self.seed = seed
        self.audio_size = audio_size
        self.build_loader()

    def build_loader(self):
        paths = []
        for path in self.root:
            paths = paths + glob(path + "/*.tfrecords")

        self.num_tfrecords=len(paths)

        print("number of tfrecord files : {}".format(self.num_tfrecords))
        
        shape = tf.stack([self.audio_size])
        filename_queue = tf.train.string_input_producer(paths, shuffle=False, seed=self.seed)

        print("mono channel audio : [batch, 16384]")
        noisy,clean = read_and_decode(filename_queue)

        clean = tf.cast(tf.reshape(clean, shape), tf.float32)
        noisy = tf.cast(tf.reshape(noisy, shape), tf.float32)
        
        min_after_dequeue = 3000
        capacity = min_after_dequeue + 3 * self.batch_size

        self.mix, self.source = tf.train.shuffle_batch([noisy, clean], batch_size=self.batch_size,num_threads=3, capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue, seed=777,name='queue')
        self.noise = self.mix-self.source
        self.zeros = tf.zeros(shape=tf.shape(self.noise))

        self.mix_queue = tf.concat([self.mix[:3*int(config.batch_size/4)],self.noise[:int(config.batch_size/4)]],axis=0)
        self.source_queue = tf.concat([self.source[:3*int(config.batch_size/4)],self.zeros[:int(config.batch_size/4)]],axis=0)
