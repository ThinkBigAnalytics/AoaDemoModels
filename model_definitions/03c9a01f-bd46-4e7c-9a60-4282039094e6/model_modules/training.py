from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.model_selection import train_test_split

import pickle
import pandas as pd
import json


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    dataset = pd.read_csv(data_conf['url'], header=None)

    # split into test and train
    train, _ = train_test_split(dataset, test_size=data_conf["test_split"], random_state=42)

    # split data into X and y
    train = train.values
    X_train = train[:, 0:8]
    y_train = train[:, 8]

    scaler = Scaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    print("Starting training...")

    # fit model to training data
    model = XGBClassifier(eta=hyperparams["eta"], max_depth=hyperparams["max_depth"])
    model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts
    pickle.dump(scaler, open("artifacts/output/scaler.pkl", "wb"))
    pickle.dump(model, open("artifacts/output/model.pkl", "wb"))

    with open("metrics/metrics.json", "w+") as f:
        json.dump({"shape": "test"}, f)

    print("Saved trained model")
