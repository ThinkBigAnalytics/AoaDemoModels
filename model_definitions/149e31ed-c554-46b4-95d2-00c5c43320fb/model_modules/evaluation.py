from pyspark import SparkConf
from pyspark.sql import SparkSession
from sklearn import metrics
from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from aoa.stats import stats
from aoa.util.artefacts import save_plot

import logging
import os
import joblib
import json
import numpy as np
import pandas as pd

logging.getLogger("py4j").setLevel(logging.ERROR)

# spark name and other properties are set by the framework launcher in spark-submit. don't override
spark = SparkSession.builder \
    .config(conf=SparkConf()) \
    .getOrCreate()


def evaluate(data_conf, model_conf, **kwargs):
    model = joblib.load('artifacts/input/model.joblib')

    # For demo purposes data is read from Vantage, but in a real environment
    # it can be anything that pyspark can read (csv, parquet, avro, etc...)
    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    # As this is for demo purposes, we simulate the test dataset changing between executions
    # by introducing a random sample. Note that the sampling is performed in Teradata!
    test_df = DataFrame(data_conf["table"]).sample(frac=0.8)
    test_pdf = test_df.to_pandas()

    X_test = test_pdf[model.feature_names]
    y_test = test_pdf[model.target_name]

    # do feature eng in spark / joins whatever reason you're using pyspark...

    print("Scoring")
    y_pred = model.predict(test_pdf[model.feature_names])

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
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

    feature_importance = pd.DataFrame(list(zip(model.feature_names, np.abs(shap_values).mean(0))),
                                      columns=['col_name', 'feature_importance_vals'])
    feature_importance = feature_importance.set_index("col_name").T.to_dict(orient='records')[0]

    stats.record_stats(test_df,
                       features=model.feature_names,
                       predictors=["HasDiabetes"],
                       categorical=["HasDiabetes"],
                       importance=feature_importance,
                       category_labels={"HasDiabetes": {0: "false", 1: "true"}})
