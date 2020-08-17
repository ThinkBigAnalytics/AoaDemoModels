import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import sklearn.datasets
import joblib
import matplotlib.pyplot as plt
import json
import shap
import lime.lime_tabular

def evaluate(data_conf, model_conf, **kwargs):
    """Python evaluate method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """

    predict_df = pd.read_csv(data_conf['location'])
    _, test = train_test_split(predict_df, test_size=0.5, random_state=42)
    X_predict = test.drop("species", 1)
    y_test = test['species']
    model = joblib.load('artifacts/input/model.joblib')
    X_train = pd.read_pickle('artifacts/input/X_train.pickle')

    y_predict = model.predict(X_predict)
    scores = {}
    scores['accuracy'] = metrics.accuracy_score(y_test, y_predict)
    print("model accuracy is ", scores['accuracy'])


    shap_explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values = shap_explainer.shap_values(X_train)

    shap.summary_plot(shap_values, X_predict, show=False)
    plt.title('SHAP diagram for feature weights')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('artifacts/output/summary.png', dpi = 100)
    scores['shap_values'] = {'Content-Type':'image/png', 'location':'S3', 'object':'summary.png'}

    # dump results as json file evaluation.json to models/ folder
    with open("metrics/metrics.json", "w+") as f:
        json.dump(scores, f)
    print("Evaluation complete...")

class ModelScorer(object):
    def __init__(self, config=None):
        self.model = joblib.load('artifacts/input/model.joblib')
        self.X_train = joblib.load('artifacts/input/X_train.pickle')
        iris = sklearn.datasets.load_iris()
        self.feature_names = iris.feature_names
        self.classes = iris.target_names
        self.explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train.values, feature_names=self.feature_names, class_names=self.classes, discretize_continuous=True)

    def predict(self, data):
        pred = self.model.predict([data])

        return pred

    def explain(self, data):
        raw_exp = self.explainer.explain_instance(pd.Series(data), self.model.predict_proba, num_features = 4, top_labels = 1).as_map()
        exp = {self.classes[k]: [(self.feature_names[x[0]],x[1]) for x in raw_exp[k]] for k in raw_exp}

        return exp


