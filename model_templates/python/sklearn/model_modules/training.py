import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import os

def train(data_conf, model_conf, **kwargs):
    """Python train method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """

    hyperparams = model_conf["hyperParameters"]

    # load data & engineer
    iris_df = pd.read_csv(data_conf['location'])
    features = 'sepallength,sepalwidth,petallength,petalwidth'.split(',')
    X = iris_df.loc[:, features]
    y = iris_df['class']

    print("Starting training...")
    # fit model to training data
    knn = KNeighborsClassifier(n_neighbors=hyperparams['n_neighbors'])
    knn.fit(X,y)
    print("Finished training")

    # export model artefacts to models/ folder
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(knn, 'models/iris_knn.joblib')
    print("Saved trained model")