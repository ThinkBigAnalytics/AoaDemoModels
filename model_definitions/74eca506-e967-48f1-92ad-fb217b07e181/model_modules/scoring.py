import json
import tensorflow as tf

from keras.models import load_model
from keras.datasets import imdb
from .preprocess import preprocess


class ModelScorer(object):
    def __init__(self, config=None):
        if not config:
            with open("config.json") as f:
                config = json.load(f)

        self.model = load_model("models/model.h5")
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

        self.max_len = config["hyperParameters"]["maxlen"]

    def predict(self, data):
        x = preprocess([data], maxlen=self.max_len)
        with self.graph.as_default():
            return self.model.predict(x)[0][0]

    def evaluate(self, x, y):
        x = preprocess(x, maxlen=self.max_len)
        metrics = self.model.evaluate(x, y)

        return {self.model.metrics_names[i]: v for i, v in enumerate(metrics)}


def evaluate(data_conf, model_conf, **kwargs):
    (_, _), (x_test, y_test) = imdb.load_data(num_words=model_conf["hyperParameters"]["max_features"])

    scorer = ModelScorer(model_conf)
    metrics = scorer.evaluate(x_test, y_test)

    with open("models/metric.json", "w+") as f:
        json.dump(metrics, f)

