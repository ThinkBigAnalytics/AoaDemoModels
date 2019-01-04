import logging

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb


def train(data_conf, model_conf):
    hyper_params = model_conf["hyperParameters"]

    logging.info('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=hyper_params["max_features"])
    logging.info('{} train sequences'.format(len(x_train)))
    logging.info('{} test sequences'.format(len(x_test)))

    logging.info('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=hyper_params["maxlen"])
    x_test = sequence.pad_sequences(x_test, maxlen=hyper_params["maxlen"])
    logging.info('x_train shape: {}'.format(x_train.shape))
    logging.info('x_test shape: {}'.format(x_test.shape))

    model = build_model_pipeline(hyper_params)

    history = model.fit(x_train, y_train,
                        batch_size=hyper_params["batch_size"],
                        epochs=hyper_params["epochs"],
                        validation_data=(x_test, y_test))

    model.save('models/model.h5')


def build_model_pipeline(hyper_params):
    logging.info('Building model...')

    model = Sequential()
    model.add(Embedding(hyper_params["max_features"],
                        hyper_params["embedding_dims"],
                        input_length=hyper_params["maxlen"]))
    model.add(Dropout(0.2))
    model.add(Conv1D(hyper_params["filters"],
                     hyper_params["kernel_size"],
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(hyper_params["hidden_dims"]))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

