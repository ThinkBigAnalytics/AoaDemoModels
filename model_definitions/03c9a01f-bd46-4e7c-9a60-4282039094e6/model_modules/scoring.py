from sklearn import metrics
from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.dataframe.copy_to import copy_to_sql

import os
import joblib
import json
import pandas as pd


def score(data_conf, model_conf, **kwargs):

    # load the model
    model = joblib.load('artifacts/input/model.joblib')

    create_context(host=data_conf["host"], username=os.environ['TD_USERNAME'], password=os.environ['TD_PASSWORD'])

    # Read test dataset from Teradata and convert to pandas.
    # As this is for demo purposes, we simulate the test dataset changing between executions
    # by introducing a random sample which can be changed in the dataset definition in the AOA
    # Note that the sampling is performed in Teradata!
    test_df = DataFrame(data_conf["table"])
    test_df = test_df.select([model.feature_names])
    test_df = test_df.sample(frac=float(data_conf["demo_sample"])).to_pandas()

    # split data into X and y
    test = test_df.values
    X_test = test[:, 0:8]
    y_test = test[:, 8]

    print("Scoring")
    y_pred = model.predict(X_test)

    print("Finished Scoring")

    # store predictions in Teradata
    y_pred = pd.DataFrame(y_pred, columns=["pred"])
    copy_to_sql(df=y_pred, table_name=data_conf["predictions"], index=True, if_exists="replace")

    return X_test, y_pred, y_test, model


def save_plot(title):
    import matplotlib.pyplot as plt

    plt.title(title)
    fig = plt.gcf()
    filename = title.replace(" ", "_").lower()
    fig.savefig('artifacts/output/{}'.format(filename), dpi=500)


def evaluate(data_conf, model_conf, **kwargs):

    X_test, y_pred, y_test, model = score(data_conf, model_conf, **kwargs)

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
