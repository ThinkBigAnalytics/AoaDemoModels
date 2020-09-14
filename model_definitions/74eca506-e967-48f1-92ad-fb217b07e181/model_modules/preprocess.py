import tensorflow as tf


def preprocess(data, maxlen):
    return tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=maxlen)