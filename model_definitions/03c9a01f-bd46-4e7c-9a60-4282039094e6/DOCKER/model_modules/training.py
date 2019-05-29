from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler as Scaler
import pickle


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    dataset = loadtxt(data_conf['data_path'], delimiter=",")

    # split data into X and y
    X_train = dataset[:, 0:8]
    y_train = dataset[:, 8]
    
    scaler = Scaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    print("Starting training...")

    # fit model to training data
    model = XGBClassifier(eta=hyperparams["eta"], max_depth=hyperparams["max_depth"])
    model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    pickle.dump(model, open("models/model.pkl", "wb"))

    print("Saved trained model")
