import logging

import tensorflow as tf

from .callback import AoaKerasProgressCallback
from .preprocess import preprocess


def train(data_conf, model_conf, **kwargs):
    hyper_params = model_conf["hyperParameters"]
    progress_callback = kwargs.get("progress_callback_handler", lambda **args: None)

    logging.info('Loading data...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=hyper_params["max_features"])

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

    model.save('artifacts/output/model.h5')


def build_model_pipeline(hyper_params):
    logging.info('Building model...')

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(hyper_params["max_features"],
                        hyper_params["embedding_dims"],
                        input_length=hyper_params["maxlen"]))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv1D(hyper_params["filters"],
                     hyper_params["kernel_size"],
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(hyper_params["hidden_dims"]))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

