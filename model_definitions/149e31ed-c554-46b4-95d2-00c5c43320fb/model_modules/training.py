from pyspark import SparkConf
from pyspark.sql import SparkSession
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from nyoka import xgboost_to_pmml
from .util import read_dataframe

import logging
import joblib
import urllib.request

logging.getLogger("py4j").setLevel(logging.ERROR)

# spark name and other properties are set by the framework launcher in spark-submit. don't override
spark = SparkSession.builder \
    .config(conf=SparkConf()) \
    .getOrCreate()


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    feature_names = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age"]
    target_name = "HasDiabetes"

    train_df = read_dataframe(spark, data_conf["url"])

    # do feature eng in spark / joins whatever reason you're using pyspark...
    # split into test and train
    train_df = train_df.randomSplit([0.7, 0.3], 42)[0].toPandas()

    # split data into X and y
    X_train = train_df.drop(target_name, 1)
    y_train = train_df[target_name]

    print("Starting training...")

    # fit model to training data
    model = Pipeline([('scaler', MinMaxScaler()),
                      ('xgb', XGBClassifier(eta=hyperparams["eta"],
                                            max_depth=hyperparams["max_depth"]))])
    # xgboost saves feature names but lets store on pipeline for easy access
    model.feature_names = feature_names

    model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts
    joblib.dump(model, "artifacts/output/model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    xgboost_to_pmml(pipeline=model, col_names=feature_names[0:8], target_name=target_name, pmml_f_name="artifacts/output/model.pmml")

    print("Saved trained model")
