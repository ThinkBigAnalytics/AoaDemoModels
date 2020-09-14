import json
import tensorflow as tf

from sklearn import metrics
from keras.models import load_model
from keras.datasets import imdb
from .preprocess import preprocess


def score(data_conf, model_conf, **kwargs):
    (_, _), (X_test, y_test) = imdb.load_data(num_words=model_conf["hyperParameters"]["max_features"])

    model = load_model("artifacts/input/model.h5")

    y_pred = model.predict(preprocess(X_test, maxlen=model_conf["hyperParameters"]["maxlen"]))

    return X_test, y_pred, y_test, model


def save_plot(title):
    import matplotlib.pyplot as plt

    plt.title(title)
    fig = plt.gcf()
    filename = title.replace(" ", "_").lower()
    fig.savefig('artifacts/output/{}'.format(filename), dpi=500)


def evaluate(data_conf, model_conf, **kwargs):
    X_test, y_pred, y_test, model = score(data_conf, model_conf, **kwargs)

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
    }

    with open("artifacts/output/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    metrics.plot_confusion_matrix(model, X_test, y_test)
    save_plot('Confusion Matrix')

    metrics.plot_roc_curve(model, X_test, y_test)
    save_plot('ROC Curve')


class ModelScorer(object):
    def __init__(self, config=None):
        if not config:
            with open("config.json") as f:
                config = json.load(f)

        self.model = load_model("artifacts/input/model.h5")
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

        self.max_len = config["hyperParameters"]["maxlen"]

    def predict(self, data):
        x = preprocess([data], maxlen=self.max_len)
        with self.graph.as_default():
            return self.model.predict(x)[0][0]
