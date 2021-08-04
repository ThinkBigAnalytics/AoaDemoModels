from sklearn import metrics
from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.dataframe.copy_to import copy_to_sql
from aoa.stats import stats
from aoa.util.artefacts import save_plot

import os
import joblib
import json
import numpy as np
import pandas as pd


def evaluate(data_conf, model_conf, **kwargs):
    model = joblib.load('artifacts/input/model.joblib')

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    # Read test dataset from Teradata
    # As this is for demo purposes, we simulate the test dataset changing between executions
    # by introducing a random sample. Note that the sampling is performed in Teradata!
    test_df = DataFrame(data_conf["table"]).sample(frac=0.8)
    test_pdf = test_df.to_pandas()

    X_test = test_pdf[model.feature_names]
    y_test = test_pdf[model.target_name]

    print("Scoring")
    y_pred = model.predict(test_pdf[model.feature_names])

    y_pred_tdf = pd.DataFrame(y_pred, columns=[model.target_name])
    y_pred_tdf["PatientId"] = test_pdf["PatientId"].values

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
                      show=False, plot_size=(12, 8), plot_type='bar')
    save_plot('SHAP Feature Importance')

    feature_importance = pd.DataFrame(list(zip(model.feature_names, np.abs(shap_values).mean(0))),
                                      columns=['col_name', 'feature_importance_vals'])
    feature_importance = feature_importance.set_index("col_name").T.to_dict(orient='records')[0]

    predictions_table = "{}_tmp".format(data_conf["predictions"]).lower()
    copy_to_sql(df=y_pred_tdf, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

    stats.record_evaluation_stats(test_df, DataFrame(predictions_table), feature_importance)
