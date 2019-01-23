from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle


def train(data_conf, model_conf, **kwargs):

    # load data
    dataset = loadtxt(data_conf['data_path'], delimiter=",")

    # split data into X and y
    X = dataset[:,0:8]
    Y = dataset[:,8]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    print("Starting training...")

    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train)

    print("Finished training")

    # save model to pickle file
    pickle.dump(model, open("models/model.pkl", "wb"))

    print("Saved trained model")