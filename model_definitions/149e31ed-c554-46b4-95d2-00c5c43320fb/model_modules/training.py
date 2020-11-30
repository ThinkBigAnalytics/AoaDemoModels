from pyspark import SparkConf
from pyspark.sql import SparkSession
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from nyoka import xgboost_to_pmml

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

    # in a real world scenario, you would read from S3, HDFS, Teradata,
    # etc but for demo reading from url. we could read via pandas.read_csv but just to show pyspark ...
    urllib.request.urlretrieve(data_conf["url"], "/tmp/data.csv")
    column_names = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age", "HasDiabetes"]
    train_df = spark.read.format("csv").load("/tmp/data.csv").toDF(*column_names)

    # do feature eng in spark / joins whatever reason you're using pyspark...
    # split into test and train
    train_df = train_df.randomSplit([0.7, 0.3], 42)[0].toPandas()

    # split data into X and y
    train = train_df.values
    X_train = train[:, 0:8]
    y_train = train[:, 8]

    print("Starting training...")

    # fit model to training data
    model = Pipeline([('scaler', MinMaxScaler()),
                      ('xgb', XGBClassifier(eta=hyperparams["eta"],
                                            max_depth=hyperparams["max_depth"]))])
    # xgboost saves feature names but lets store on pipeline for easy access
    model.feature_names = column_names[0:8]

    model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts
    joblib.dump(model, "artifacts/output/model.joblib")
    xgboost_to_pmml(pipeline=model, col_names=column_names[0:8], target_name="HasDiabetes", pmml_f_name="artifacts/output/model.pmml")

    print("Saved trained model")
