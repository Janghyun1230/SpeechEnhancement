import numpy as numpy
import tensorflow as tf
from configuration import get_config

config = get_config()


# list를 받아서 tf feature 꼴로 변환
# https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/core/example/feature.proto
# tf.train.Feature로 변환하고 tf.train.Features로 묶음

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten().tolist()))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(audio, filename):
    # audio input [source, time segments(stride 1/2), data]
    num_segments = audio.shape[1]

    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_segments):
        example = tf.train.Example(features=tf.train.Features(feature={
            'noisy': _float_feature(audio[0, index, :]),
            'clean': _float_feature(audio[1, index, :]),
            }))
        writer.write(example.SerializeToString())

    writer.close()


# Remember to generate a file name queue of you 'train.TFRecord' file path
def read_and_decode_music(filename_queue, audio_size= config.audio_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'mixture': tf.FixedLenFeature([audio_size], tf.float32),
            'drums': tf.FixedLenFeature([audio_size], tf.float32),
            'bass': tf.FixedLenFeature([audio_size], tf.float32),
            'others': tf.FixedLenFeature([audio_size], tf.float32),
            'vocal': tf.FixedLenFeature([audio_size], tf.float32)
            })

    mixture = features['mixture']
    drums = features['drums']
    bass = features['bass']
    others = features['others']
    vocal = features['vocal']

    return mixture, bass, drums, others, vocal


# Remember to generate a file name queue of you 'train.TFRecord' file path
def read_and_decode(filename_queue, audio_size= config.audio_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'noisy': tf.FixedLenFeature([audio_size], tf.float32),
            'clean': tf.FixedLenFeature([audio_size], tf.float32),
            })

    noisy = features['noisy']
    clean = features['clean']

    return noisy, clean
