from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import json


class ModelScorer(object):
    def __init__(self, config=None):
        self.model = pickle.load(open("models/model.pkl", 'rb'))

    def predict(self, data):
        return self.model.predict([data])

    def evaluate(self, x, y):
        y_pred = self.model.predict(x)

        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y, predictions)

        return {'accuracy': (accuracy * 100.0)}


def evaluate(data_conf, model_conf):

    # load data
    dataset = loadtxt(data_conf['data_path'], delimiter=",")

    # split data into X and y
    X = dataset[:,0:8]
    Y = dataset[:,8]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    scorer = ModelScorer(model_conf)
    scores = scorer.evaluate(X_test, y_test)

    print(scores)

    with open("models/evaluation.json", "w+") as f:
        json.dump(scores, f)
