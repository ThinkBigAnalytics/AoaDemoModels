import json

from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.analytics.sqle import GLMPredict
from teradataml.options.display import display
import os

display.print_sqlmr_query = True


def evaluate(data_conf, model_conf, **kwargs):

    create_context(host = data_conf["hostname"],
                   username = os.environ['TD_USERNAME'],
                   password = os.environ['TD_PASSWORD'])

    model = DataFrame(data_conf["model_table"])

    dataset = DataFrame(data_conf["data_table"])

    print("Starting Evaluation...")

    predicted = GLMPredict(modeldata = model,
                               newdata = dataset,
                               terms = model_conf["terms"],
                               family = model_conf["family"],
                               linkfunction = model_conf["linkfunction"])

    predicted.result.to_sql(data_conf["prediction_table"], if_exists="replace")

    print("Finished Evaluation")

    true_label = model_conf["terms"]
    df_pred = DataFrame(data_conf["prediction_table"])
    df_pred = df_pred.assign(err = df_pred[true_label] - df_pred.fitted_value)
    df_pred = df_pred.assign(err_squared = df_pred.err * df_pred.err)

    mean_err_squared = df_pred.agg({"err_squared":"mean"}).to_pandas()

    print("mean squared error: ", str(mean_err_squared['mean_err_squared']))

    with open("models/evaluation.json", "w+") as f:
        json.dump({'mean_squared_err': str(mean_err_squared['mean_err_squared'][0])}, f)

    print("Saved evaluation results")
