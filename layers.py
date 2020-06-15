import tensorflow as tf
import numpy as np
from Utils import *
from tensorflow.python.layers import utils


def conv1d(name, input, oc, f_w=3, s_w=1, d_w=1, bn=True, is_training=True, print_shape=False, act='lrelu'):
    """
    1d convolution module (conv-bn-act)

    Args:
        input: NWC
    """

    with tf.variable_scope(name) as scope:
        if print_shape:
            print("{} shape : {}".format(input.name, input.get_shape()))

        input_shape = input.get_shape().as_list()
        filter = tf.get_variable("filter", shape=[f_w, input_shape[-1], oc],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.convolution(input, filter, strides=[s_w], dilation_rate=[d_w],
                                 padding="SAME", name="1d_conv")

        if bn:
            bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, renorm=True, scope='bn')
        else:
            bn = conv

        if act == 'relu':
            act = tf.nn.relu(bn)
        elif act == 'lrelu':
            act = tf.nn.leaky_relu(bn)
        else:
            act = bn

    return act


def deconv1d(name, input, oc, f_w=3, s_w=1, bn=True, is_training=True, print_shape=False, act='lrelu'):
    """
    2d transpose convolution module (conv-bn-act)

    Args:
        input: NHWC

    """
    with tf.variable_scope(name) as scope:
        if print_shape:
            print("{} shape : {}".format(input.name, input.get_shape()))

        input_shape = input.get_shape().as_list()
        output_shape = (input_shape[0], config.audio_size, oc)

        filter = tf.get_variable("filter", shape=[f_w, oc, input_shape[-1]],
                            initializer=tf.contrib.layers.xavier_initializer())

        deconv = tf.contrib.nn.conv1d_transpose(input, filter=filter, output_shape=output_shape, stride=s_w,
                                        padding="SAME", name="1d_transposeconv")

        if bn:
            bn = tf.contrib.layers.batch_norm(deconv, center=True, scale=True, renorm=True, scope='bn')
        else:
            bn = deconv

        if act == 'relu':
            act = tf.nn.relu(bn)
        elif act == 'lrelu':
            act = tf.nn.leaky_relu(bn)
        else:
            act = bn

        return act


def conv2d_real(name, input, oc, f_h=3, f_w=3, s_h=1, s_w=1, bn=True, is_training=True, print_shape=False, act='lrelu'):
    with tf.variable_scope(name) as scope:
        input_shape = input.get_shape().as_list()
        W = tf.get_variable("W", shape=[f_h, f_w, input_shape[-1], oc],
                            initializer=tf.contrib.layers.xavier_initializer())

        conv = tf.nn.conv2d(input, W, strides=[1, s_h, s_w, 1], padding="SAME", name="conv")

        if bn:
            bn = tf.contrib.layers.batch_norm(conv,
                                          center=True, scale=True,
                                          renorm=True,
                                          scope='bn')
        else:
            bn = conv

        if act == 'relu':
            # This is regular relu
            act = tf.nn.relu(bn)
        elif act == 'lrelu':
            # This is leaky relu
            act = tf.nn.leaky_relu(bn)
        else:
            print("linear activation")
            act = bn

        if print_shape:
            print("{} shape : {}".format(act.name, act.get_shape()))

        return act


def deconv2d_real(name, input, oc, f_h=3, f_w=3, s_h=1, s_w=1, bn=True, is_training=True, print_shape=False, act='lrelu'):
    with tf.variable_scope(name) as scope:
        padding = "SAME"
        input_shape = input.get_shape().as_list()
        out_h = utils.deconv_output_length(input_shape[1], f_h, padding, s_h)
        out_w = utils.deconv_output_length(input_shape[2], f_w, padding, s_w)
        output_shape = (input_shape[0], out_h, out_w, oc)
        strides = [1, s_h, s_w, 1]

        W = tf.get_variable("W", shape=[f_h, f_w, oc, input_shape[-1]], initializer=tf.contrib.layers.xavier_initializer())

        deconv = tf.nn.conv2d_transpose(input, filter=W, output_shape=output_shape, strides=strides, padding=padding)

        if bn:
            bn = tf.contrib.layers.batch_norm(deconv,
                                          center=True, scale=True,
                                          renorm=True,
                                          scope='bn')
        else:
            bn = deconv

        if act == 'relu':
            # Just use the regular relu
            act = tf.nn.relu(bn)
        elif act == 'lrelu':
            # This is leaky relu
            act = tf.nn.leaky_relu(bn)
        elif act == 'abs':
            act = tf.abs(bn)
        else:
            act = bn

        if print_shape:
            print("{} shape : {}".format(act.name, act.get_shape()))

        return act

if  __name__ == "__main__":
    tf.reset_default_graph()
    input = tf.constant(np.expand_dims(np.expand_dims(np.array([[1, 2], [2, 1]]), axis=-1),axis=-1), dtype=tf.float32)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("input")
        print(sess.run(input))
