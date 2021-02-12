from pyspark import SparkConf
from pyspark.sql import SparkSession
from .util import read_dataframe

import logging
import joblib
import pandas as pd
import os


logging.getLogger("py4j").setLevel(logging.ERROR)

# spark name and other properties are set by the framework launcher in spark-submit. don't override
spark = SparkSession.builder \
    .config(conf=SparkConf()) \
    .getOrCreate()


def score(data_conf, model_conf, **kwargs):
    model = joblib.load('artifacts/input/model.joblib')

    test_df = read_dataframe(spark, data_conf["url"])

    # do feature eng in spark / joins whatever reason you're using pyspark...
    # split into test and train
    test_df = test_df.randomSplit([0.7, 0.3], 42)[1].toPandas()

    X = test_df[model.feature_names]

    print("Scoring")
    y_pred = model.predict(X)

    y_pred = pd.DataFrame(y_pred, columns=["pred"])

    # wrap as pyspark df
    predictions = spark.createDataFrame(y_pred)

    # in a real world you would write the results back to HDFS, Teradata, S3 etc.
    predictions.write.mode("overwrite").save("/tmp/predictions")

    logging.info("Finished scoring")


# Add code required for RESTful API
class ModelScorer(object):

    def __init__(self, config=None):
        self.model = joblib.load('artifacts/input/model.joblib')

        from prometheus_client import Counter
        self.pred_class_counter = Counter('model_prediction_classes',
                                          'Model Prediction Classes', ['model', 'version', 'clazz'])

    def predict(self, data):
        pred = self.model.predict([data])

        # record the predicted class so we can check model drift (via class distributions)
        self.pred_class_counter.labels(model=os.environ["MODEL_NAME"],
                                       version=os.environ.get("MODEL_VERSION", "1.0"),
                                       clazz=str(int(pred))).inc()

        return pred
