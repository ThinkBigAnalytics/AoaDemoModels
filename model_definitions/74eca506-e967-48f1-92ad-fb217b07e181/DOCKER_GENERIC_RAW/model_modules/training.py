import logging

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.callbacks import Callback
from .preprocess import preprocess


class AoaKerasProgressCallback(Callback):

    def __init__(self, aoa_progress_callback):
        self.aoa_progress_callback = aoa_progress_callback
        self.iter = 0
        self.epoc = 0
        self.score = 0

    def on_epoch_end(self, epoc, logs={}):
        self.epoc = epoc
        self.score = logs.get("acc", 0)
        self._send_callback()

    def on_batch_end(self, batch, logs={}):
        self.iter = batch
        self.score = logs.get("acc", 0)
        self._send_callback()

    def _send_callback(self):
        self.aoa_progress_callback(iter=self.iter, epoc=self.epoc, score=self.score.item())


def train(data_conf, model_conf, **kwargs):
    hyper_params = model_conf["hyperParameters"]
    progress_callback = kwargs.get("progress_callback_handler", lambda *args: None)

    logging.info('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=hyper_params["max_features"])
    logging.info('{} train sequences'.format(len(x_train)))
    logging.info('{} test sequences'.format(len(x_test)))

    logging.info('Pad sequences (samples x time)')
    x_train = preprocess(x_train, maxlen=hyper_params["maxlen"])
    x_test = preprocess(x_test, maxlen=hyper_params["maxlen"])
    logging.info('x_train shape: {}'.format(x_train.shape))
    logging.info('x_test shape: {}'.format(x_test.shape))

    model = build_model_pipeline(hyper_params)

    history = model.fit(x_train, y_train,
                        batch_size=hyper_params["batch_size"],
                        epochs=hyper_params["epochs"],
                        validation_data=(x_test, y_test),
                        callbacks=[AoaKerasProgressCallback(progress_callback)])

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

