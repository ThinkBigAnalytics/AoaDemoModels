import json
import tensorflow as tf
import numpy as np

from sklearn import metrics
from .preprocess import preprocess


def score(data_conf, model_conf, **kwargs):
    hyper_params = model_conf["hyperParameters"]

    (_, _), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=hyper_params["max_features"])

    model = tf.keras.models.load_model("artifacts/input/model.h5")

    X_test = preprocess(X_test, maxlen=model_conf["hyperParameters"]["maxlen"])
    y_pred = model.predict(X_test)

    return X_test, y_pred, y_test, model


def evaluate(data_conf, model_conf, **kwargs):
    X_test, y_pred, y_test, model = score(data_conf, model_conf, **kwargs)
    y_pred = np.argmax(y_pred, axis=1)

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
    }

    with open("artifacts/output/metrics.json", "w+") as f:
        json.dump(evaluation, f)


class ModelScorer(object):
    def __init__(self, config=None):
        if not config:
            with open("config.json") as f:
                config = json.load(f)

        self.model = tf.keras.models.load_model("artifacts/input/model.h5")
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

        self.max_len = config["hyperParameters"]["maxlen"]

    def predict(self, data):
        x = preprocess([data], maxlen=self.max_len)
        with self.graph.as_default():
            return self.model.predict(x)[0][0]
