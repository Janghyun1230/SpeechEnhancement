import tensorflow as tf
import numpy as np
from configuration import config


def loss_fn(spec_est, wav_est, spec_tgt, wav_tgt, noise_tgt, loss_type=config.loss):
    if loss_type == "spec":
        loss = tf.reduce_sum(tf.reduce_mean(tf.square(spec_est - spec_tgt), axis=0))

    elif loss_type == "l2":
        noise_est = wav_tgt + noise_tgt - wav_est

        loss = tf.reduce_sum(tf.reduce_mean(tf.square(wav_est - wav_tgt), axis=0)) + \
               tf.reduce_sum(tf.reduce_mean(tf.square(noise_est - noise_tgt), axis=0))

    elif loss_type == "l1":
        noise_est = wav_tgt + noise_tgt - wav_est

        loss = tf.reduce_sum(tf.reduce_mean(tf.abs(wav_est - wav_tgt), axis=0)) + \
               tf.reduce_sum(tf.reduce_mean(tf.abs(noise_est - noise_tgt), axis=0))

    elif loss_type == "sdr":
        noise_est = wav_tgt + noise_tgt - wav_est
        s_target = tf.reduce_sum(wav_est * wav_tgt, axis=1, keepdims=True) / (tf.reduce_sum(wav_tgt**2,axis=1, keepdims=True)+1e-7) * wav_tgt
        e_noise = wav_est - s_target

        loss = - tf.reduce_mean(tf.log(tf.reduce_sum(tf.square(s_target), axis=1) + 1e-6) -
                              tf.log(tf.reduce_sum(tf.square(e_noise), axis=1) + 1e-6))

    elif loss_type == "wsdr":
        noise_est = wav_tgt + noise_tgt - wav_est

        clean_coef = tf.reduce_sum(tf.square(wav_tgt), axis=1)
        noise_coef = tf.reduce_sum(tf.square(noise_tgt), axis=1)

        coef_sum = clean_coef+noise_coef
        clean_coef_mask = clean_coef/(coef_sum+1e-7)
        noise_coef_mask = noise_coef/(coef_sum+1e-7)

        den1 = tf.sqrt(tf.reduce_sum(wav_est**2,axis=1) * tf.reduce_sum(wav_tgt**2,axis=1)+1e-7)
        num1 = tf.reduce_sum(wav_est * wav_tgt, axis=1)
        den2 = tf.sqrt(tf.reduce_sum(noise_est**2,axis=1) * tf.reduce_sum(noise_tgt**2,axis=1)+1e-7)
        num2 = tf.reduce_sum(noise_est * noise_tgt, axis=1)

        loss = -1*tf.reduce_mean(clean_coef_mask*(num1/(den1+1e-7)),axis=0) \
               -1*tf.reduce_mean(noise_coef_mask*(num2/(den2+1e-7)),axis=0)

    elif loss_type == "mix":
        loss1 = tf.reduce_sum(tf.reduce_mean(tf.square(spec_est - spec_tgt), axis=0))

        noise_est = wav_tgt + noise_tgt - wav_est

        noise_est = wav_tgt + noise_tgt - wav_est

        loss2 = tf.reduce_sum(tf.reduce_mean(tf.square(wav_est - wav_tgt), axis=0)) + \
               tf.reduce_sum(tf.reduce_mean(tf.square(noise_est - noise_tgt), axis=0))

        loss = (loss1 + loss2)/2

    else:
        raise AssertionError("wrong loss type !!")

    return loss

