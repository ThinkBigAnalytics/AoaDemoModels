from keras.preprocessing import sequence


def preprocess(data, maxlen):
    return sequence.pad_sequences(data, maxlen=maxlen)