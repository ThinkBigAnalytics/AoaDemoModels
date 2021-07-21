import sklearn.metrics as skm
from teradataml import create_context, remove_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.dataframe.copy_to import copy_to_sql
from aoa.stats import stats
from aoa.util.artefacts import save_plot

import os
import joblib
import json
import numpy as np
import pandas as pd


def save_plot(title):
    import matplotlib.pyplot as plt

    plt.title(title)
    fig = plt.gcf()
    filename = title.replace(" ", "_").lower()
    fig.savefig('artifacts/output/{}'.format(filename), dpi=500)
    plt.clf()


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
    test_pdf = test_df.to_pandas(all_rows = True)
    
    X_test = test_pdf[model.feature_names]
    y_test = test_pdf[model.target_name[0]]

    print("Scoring")
    y_pred = model.predict(X_test)
    y_pred_tdf = pd.DataFrame(y_pred, columns=model.target_name)

    evaluation = {
        'R-Squared': '{:.2f}'.format(skm.r2_score(y_test, y_pred)),
        'MAE': '{:.2f}'.format(skm.mean_absolute_error(y_test, y_pred)),
        'MSE': '{:.2f}'.format(skm.mean_squared_error(y_test, y_pred)),
        'MSLE': '{:.2f}'.format(skm.mean_squared_log_error(y_test, y_pred))
    }

    print("Saving metrics to artifacts/output/metrics.json")
    with open("artifacts/output/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    print("Saving plots")
    # a plot of actual num_orders vs predicted
    result_df = pd.DataFrame(np.vstack((y_test, y_pred)).T, columns=['Actual', 'Predicted'])
    df = result_df.sample(n=100, replace=True)
    df['No.'] = range(len(df))
    df.plot(x="No.", y=['Actual', 'Predicted'], kind = 'line', legend=True,
         subplots = False, sharex = True, figsize = (5.5,4), ls="none",
         marker="o", alpha=0.4)
    save_plot('Actual vs Predicted')

    # randomforestregressor has its own feature importance plot support
    # but lets use shap as explainability example
    import shap
    mapper = model['mapper']
    shap_explainer = shap.TreeExplainer(model['regressor'])
    X_test = pd.DataFrame(mapper.transform(X_test), columns=model.feature_names_tr)
    X_shap = shap.sample(X_test, 100)
    shap_values = shap_explainer.shap_values(X_shap)
    shap.summary_plot(shap_values, X_test, feature_names=model.feature_names_tr,
                   show=False, plot_size=(12, 8), plot_type='bar')
    save_plot('SHAP Feature Importance')
    print("Saving stats")
    feature_importances = pd.DataFrame(list(zip(model.feature_names_tr,
                                             np.abs(shap_values).mean(0))),
                                   columns=['col_name', 'feature_importance_vals'])
    feature_importances = feature_importances.set_index("col_name").T.to_dict(orient='records')[0]

    predictions_table="{}_tmp".format(data_conf["table"]).lower()
    copy_to_sql(df=y_pred_tdf, table_name=predictions_table, index=False, if_exists="replace", temporary=True)
    stats.record_evaluation_stats(test_df, DataFrame(predictions_table), feature_importances)
    remove_context()
    print("All done!")
