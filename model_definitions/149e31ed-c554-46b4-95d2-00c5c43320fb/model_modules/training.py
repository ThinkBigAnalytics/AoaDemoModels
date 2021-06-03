from pyspark import SparkConf
from pyspark.sql import SparkSession
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from nyoka import xgboost_to_pmml
from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from aoa.stats import stats
from aoa.util.artefacts import save_plot

import logging
import joblib
import os

logging.getLogger("py4j").setLevel(logging.ERROR)

# spark name and other properties are set by the framework launcher in spark-submit. don't override
spark = SparkSession.builder \
    .config(conf=SparkConf()) \
    .getOrCreate()


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    # For demo purposes we read data from Vantage, but in a real environment
    # it can be anything that pyspark can read (csv, parquet, avro, etc...)
    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    feature_names = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age"]
    target_name = "HasDiabetes"

    # read training dataset from Teradata
    train_df = DataFrame(data_conf["table"])
    train_df = train_df.select([feature_names + [target_name]])
    train_pdf = train_df.to_pandas()

    # do feature eng in spark / joins whatever reason you're using pyspark...

    # split data into X and y
    X_train = train_pdf.drop(target_name, 1)
    y_train = train_pdf[target_name]

    print("Starting training...")

    # fit model to training data
    model = Pipeline([('scaler', MinMaxScaler()),
                      ('xgb', XGBClassifier(eta=hyperparams["eta"],
                                            max_depth=hyperparams["max_depth"]))])
    # xgboost saves feature names but lets store on pipeline for easy access
    model.feature_names = feature_names
    model.target_name = target_name

    model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts
    joblib.dump(model, "artifacts/output/model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    xgboost_to_pmml(pipeline=model, col_names=feature_names[0:8], target_name=target_name, pmml_f_name="artifacts/output/model.pmml")

    print("Saved trained model")

    from xgboost import plot_importance
    model["xgb"].get_booster().feature_names = feature_names
    plot_importance(model["xgb"].get_booster(), max_num_features=10)
    save_plot("feature_importance.png")

    feature_importance = model["xgb"].get_booster().get_score(importance_type="weight")
    stats.record_stats(train_df,
                       features=feature_names,
                       predictors=["HasDiabetes"],
                       categorical=["HasDiabetes"],
                       importance=feature_importance,
                       category_labels={"HasDiabetes": {0: "false", 1: "true"}})
