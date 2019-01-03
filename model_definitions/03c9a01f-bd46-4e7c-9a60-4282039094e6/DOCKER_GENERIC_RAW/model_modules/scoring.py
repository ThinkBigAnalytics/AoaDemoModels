# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import json


def evaluate(data_conf, model_conf):

    # load data
    dataset = loadtxt(data_conf['dataset_file_path'], delimiter=",")

    # split data into X and y
    X = dataset[:,0:8]
    Y = dataset[:,8]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # load trained model
    model = pickle.load(open("models/model.pkl", 'rb'))

    # evaluate
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # dump evaluation scores into JSON
    scores = {'accuracy': (accuracy * 100.0)}

    print(scores['accuracy'])

    with open("models/evaluation.json", "w+") as f:
        json.dump(scores, f)
