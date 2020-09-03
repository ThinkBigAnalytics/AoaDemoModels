from sklearn import metrics
from sklearn.model_selection import train_test_split

import os
import joblib
import json
import pandas as pd
import shap
import matplotlib.pyplot as plt


def score(data_conf, model_conf, **kwargs):
    dataset = pd.read_csv(data_conf['url'], header=None)

    # in a real world scenario - the scoring will ONLY take a dataset to score and NOT split it like this
    # but for demo model purposes with a simple simple dataset, lets split
    _, test = train_test_split(dataset, test_size=data_conf["test_split"], random_state=42)

    # split data into X and y
    test = test.values
    X_test = test[:, 0:8]
    y_test = test[:, 8]

    model = joblib.load('models/model.joblib')
    y_pred = model.predict(X_test)

    print("Finished Scoring")

    # store predictions somewhere.. As this is demo, we'll just print to stdout.
    print(y_pred)

    return X_test, y_pred, y_test, model


def save_plot(title):
    plt.title(title)
    fig = plt.gcf()
    filename = title.replace(" ", "_").lower()
    fig.savefig('artifacts/output/{}'.format(filename), dpi=500)


def evaluate(data_conf, model_conf, **kwargs):

    X_test, y_pred, y_test, model = score(data_conf, model_conf, **kwargs)

    evaluation = {
        'Accuracy': metrics.accuracy_score(y_test, y_pred),
        'Recall': metrics.recall_score(y_test, y_pred),
        'Precision': metrics.precision_score(y_test, y_pred),
        'f1-score': metrics.f1_score(y_test, y_pred)
    }

    with open("metrics/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    metrics.plot_confusion_matrix(model, X_test, y_test)
    save_plot('Confusion Matrix')

    metrics.plot_roc_curve(model, X_test, y_test)
    save_plot('ROC Curve')

    # xgboost has its own feature importance plot support but lets use shap as explainability example
    shap_explainer = shap.TreeExplainer(model['xgb'])
    shap_values = shap_explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, feature_names=model.feature_names, show=False)
    save_plot('SHAP Feature Importance')


# Add code required for RESTful API
class ModelScorer(object):

    def __init__(self, config=None):
        self.model = joblib.load('models/model.joblib')

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
