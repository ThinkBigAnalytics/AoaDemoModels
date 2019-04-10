from numpy import loadtxt
from sklearn.metrics import accuracy_score
import pickle
import json


class ModelScorer(object):
    def __init__(self, config=None):
        self.scaler = pickle.load(open("models/scaler.pkl", 'rb'))
        self.model = pickle.load(open("models/model.pkl", 'rb'))

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

    dataset = loadtxt(data_conf['data_path'], delimiter=",")

    # split data into X and y
    X_test = dataset[:, 0:8]
    y_test = dataset[:, 8]

    scorer = ModelScorer(model_conf)
    scores = scorer.evaluate(X_test, y_test)

    print(scores)

    with open("models/evaluation.json", "w+") as f:
        json.dump(scores, f)
