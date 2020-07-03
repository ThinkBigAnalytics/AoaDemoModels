from sklearn import metrics
from sklearn.model_selection import train_test_split

import joblib
import json
import pandas as pd


def score(data_conf, model_conf, **kwargs):
    dataset = pd.read_csv(data_conf['url'], header=None)

    # in a real world scenario - the scoring will ONLY take a dataset to score and NOT split it like this
    # but for demo model purposes with a simple simple dataset, lets split
    _, test = train_test_split(dataset, test_size=data_conf["test_split"], random_state=42)

    # split data into X and y
    test = test.values
    X_test = test[:, 0:8]
    y_test = test[:, 8]

    clf = joblib.load('models/model.joblib')

    y_pred = clf.predict(X_test)

    print("Finished Scoring")

    # store predictions somewhere.. As this is demo, we'll just print to stdout.
    print(y_pred)

    return y_pred, y_test


def evaluate(data_conf, model_conf, **kwargs):

    y_pred, y_test = score(data_conf, model_conf, **kwargs)

    evaluation = {
        'Accuracy': metrics.accuracy_score(y_test, y_pred),
        'Recall': metrics.recall_score(y_test, y_pred),
        'Precision': metrics.precision_score(y_test, y_pred),
        'f1-score': metrics.f1_score(y_test, y_pred)
    }

    print("model metrics: {}".format(evaluation))

    with open("models/evaluation.json", "w+") as f:
        json.dump(evaluation, f)


# Add code required for RESTful API
class ModelScorer(object):
    def __init__(self, config=None):
        self.model = joblib.load('models/model.joblib')

        from prometheus_client import Counter
        self.pred_class_counter = Counter('model_prediction_classes',
                                          'Model Prediction Classes', ['model', 'version', 'clazz'])

    def record_prediction_stats(self, pred):
        import os
        self.pred_class_counter.labels(model=os.environ["MODEL_NAME"],
                                       version=os.environ.get("MODEL_VERSION", "1.0"),
                                       clazz=str(int(pred))).inc()

    def predict(self, data):
        pred = self.model.predict([data])

        self.record_prediction_stats(pred)

        return pred
