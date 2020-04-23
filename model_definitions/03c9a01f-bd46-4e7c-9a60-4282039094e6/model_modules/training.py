from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.model_selection import train_test_split
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml as dump_pmml

import pickle
import pandas as pd


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
    classifier = XGBClassifier(eta=hyperparams["eta"], max_depth=hyperparams["max_depth"])

    pipeline = PMMLPipeline([
        ("scaler", scaler),
        ("classifier", classifier)
    ])

    print("Starting training...")

    # fit model to training data
    pipeline.fit(X_train, y_train)

    print("Finished training")

    # export model artifacts
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    pickle.dump(classifier, open("models/model.pkl", "wb"))
    dump_pmml(pipeline, "models/pipeline.pmml", debug=True)

    print("Saved trained model")
