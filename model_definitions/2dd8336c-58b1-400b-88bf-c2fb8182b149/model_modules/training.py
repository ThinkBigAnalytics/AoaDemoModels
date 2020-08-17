import pandas as pd
import sklearn.ensemble
from sklearn.model_selection import train_test_split
import joblib
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
    train, _ = train_test_split(iris_df, test_size=0.5, random_state=42)
    X = train.drop("species", 1)
    y = train['species']

    print("Starting training...")
    # fit model to training data
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=hyperparams['n_estimators'])
    model.fit(X,y)
    print("Finished training")

    # export model artefacts to models/ folder
    if not os.path.exists('artifacts/output'):
        os.makedirs('artifacts/output')
    joblib.dump(model, 'artifacts/output/model.joblib')
    X.to_pickle('artifacts/output/X_train.pickle') # Saving training set for future explainers
    print("Saved trained model")