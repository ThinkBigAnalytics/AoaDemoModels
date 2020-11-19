from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml
from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame

import joblib
import os


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    create_context(host=data_conf["host"], username=os.environ['TD_USERNAME'], password=os.environ['TD_PASSWORD'])

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame(data_conf["table"])
    train_df = train_df.select([["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age", "HasDiabetes"]])
    train_df = train_df.to_pandas()

    # split data into X and y
    train = train_df.values
    X_train = train[:, 0:8]
    y_train = train[:, 8]

    print("Starting training...")

    # fit model to training data
    model = Pipeline([('scaler', MinMaxScaler()),
                     ('xgb', XGBClassifier(eta=hyperparams["eta"],
                                           max_depth=hyperparams["max_depth"]))])
    # xgboost saves feature names but lets store on pipeline for easy access later
    model.feature_names = train_df.columns.tolist()

    model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts
    joblib.dump(model, "artifacts/output/model.joblib")
    sklearn2pmml(make_pmml_pipeline(model), "artifacts/output/model.pmml")

    print("Saved trained model")
