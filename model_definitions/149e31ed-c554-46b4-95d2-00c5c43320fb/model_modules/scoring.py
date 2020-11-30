from pyspark import SparkConf
from pyspark.sql import SparkSession
from sklearn import metrics

import logging
import joblib
import urllib.request
import pandas as pd
import json
import os


logging.getLogger("py4j").setLevel(logging.ERROR)

# spark name and other properties are set by the framework launcher in spark-submit. don't override
spark = SparkSession.builder \
    .config(conf=SparkConf()) \
    .getOrCreate()


def read_dataframe(url):
    # in a real world scenario, you would read from S3, HDFS, Teradata,
    # etc but for demo reading from url. we could read via pandas.read_csv but just to show pyspark ...
    urllib.request.urlretrieve(url, "/tmp/data.csv")
    column_names = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age", "HasDiabetes"]

    return spark.read.format("csv").load("/tmp/data.csv").toDF(*column_names)


def score(data_conf, model_conf, **kwargs):

    # load the model
    model = joblib.load('artifacts/input/model.joblib')

    test_df = read_dataframe(data_conf["url"])

    # do feature eng in spark / joins whatever reason you're using pyspark...
    # split into test and train
    test_df = test_df.randomSplit([0.7, 0.3], 42)[1].toPandas()

    X_test = test_df[model.feature_names]

    print("Scoring")
    y_pred = model.predict(X_test)

    y_pred = pd.DataFrame(y_pred, columns=["pred"])

    # wrap as pyspark df
    predictions = spark.createDataFrame(y_pred)

    # in a real world you would write the results back to HDFS, Teradata, S3 etc.
    predictions.write.mode("overwrite").save("/tmp/predictions")

    logging.info("Finished scoring")

    return X_test, y_pred, model


def save_plot(title):
    import matplotlib.pyplot as plt

    plt.title(title)
    fig = plt.gcf()
    filename = title.replace(" ", "_").lower()
    fig.savefig('artifacts/output/{}'.format(filename), dpi=500)
    plt.clf()


def evaluate(data_conf, model_conf, **kwargs):

    X_test, y_pred, model = score(data_conf, model_conf, **kwargs)

    # again, in a real world you would read from S3, HDFS, Teradata and join the y_pred index to read the expected
    # values for the dataset you scored
    y_test = read_dataframe(data_conf["url"]).select("HasDiabetes")
    y_test = y_test.randomSplit([0.7, 0.3], 42)[1].toPandas()

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred, pos_label='1')),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred, pos_label='1')),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred, pos_label='1'))
    }

    with open("artifacts/output/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    metrics.plot_confusion_matrix(model, X_test, y_test)
    save_plot('Confusion Matrix')

    metrics.plot_roc_curve(model, X_test, y_test)
    save_plot('ROC Curve')

    # xgboost has its own feature importance plot support but lets use shap as explainability example
    import shap

    shap_explainer = shap.TreeExplainer(model['xgb'])
    shap_values = shap_explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, feature_names=model.feature_names,
                      show=False, plot_size=(12,8), plot_type='bar')
    save_plot('SHAP Feature Importance')


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
