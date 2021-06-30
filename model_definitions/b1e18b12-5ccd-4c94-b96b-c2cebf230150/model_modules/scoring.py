from teradataml import create_context, remove_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.dataframe.copy_to import copy_to_sql
from aoa.stats import stats

import os
import joblib
import pandas as pd


def score(data_conf, model_conf, **kwargs):
    model = joblib.load("artifacts/input/model.joblib")

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    features_tdf = DataFrame(data_conf["table"])
    # convert to pandas to use locally
    features_df = features_tdf.to_pandas(all_rows = True)

    print("Scoring")
    y_pred = model.predict(features_df[model.feature_names])

    print("Finished Scoring")

    # create result dataframe and store in Teradata
    y_pred = pd.DataFrame(y_pred, columns=["pred"])
    copy_to_sql(df=y_pred, table_name=data_conf["predictions"], index=False, if_exists="replace")

    # send statistics for monitoring
    predictions_tdf = DataFrame(data_conf["predictions"])
    stats.record_scoring_stats(features_tdf, predictions_tdf, mapper={model.target_name[0]: "pred"})

    remove_context()


# Add code required for RESTful API
class ModelScorer(object):

    def __init__(self, config=None):
        self.model = joblib.load('artifacts/input/model.joblib')

        from prometheus_client import Counter
        self.pred_class_counter = Counter('model_prediction_classes',
                                          'Model Prediction Classes', 
                                          ['model', 'version', 'clazz'])

    def predict(self, data):

        data_df=pd.DataFrame([data], columns=self.model.feature_names)
        pred = self.model.predict(data_df)

        # record the predicted class so we can check model drift (via class distributions)
        self.pred_class_counter.labels(model=os.environ["MODEL_NAME"],
                                       version=os.environ.get("MODEL_VERSION",
                                                              "1.0"),
                                       clazz=str(pred)).inc()

        return pred
