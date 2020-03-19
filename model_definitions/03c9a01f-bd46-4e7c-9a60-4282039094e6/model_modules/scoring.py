from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pickle
import json
import pandas as pd


class ModelScorer(object):
    def __init__(self, config=None):
        self.scaler = pickle.load(open("artifacts/input/scaler.pkl", 'rb'))
        self.model = pickle.load(open("artifacts/input/model.pkl", 'rb'))

    def predict(self, data):
        data = self.scaler.transform([data])
        return self.model.predict(data)

    def evaluate(self, x, y):
        x = self.scaler.transform(x)
        y_pred = self.model.predict(x)

        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y, predictions)

        return {'accuracy': (accuracy * 100.0)}


def evaluate(data_conf, model_conf, **kwargs):

    dataset = pd.read_csv(data_conf['url'], header=None)

    # split into test and train
    _, test = train_test_split(dataset, test_size=data_conf["test_split"], random_state=42)

    print(test.shape)

    # split data into X and y
    test = test.values
    X_test = test[:, 0:8]
    y_test = test[:, 8]

    scorer = ModelScorer(model_conf)
    scores = scorer.evaluate(X_test, y_test)

    with open("metrics/metrics.json", "w+") as f:
        json.dump(scores, f)

    model = pickle.load(open("input/model.pkl", 'rb'))
    pickle.dump(model, open("artifacts/output/eval.pkl", "wb"))

