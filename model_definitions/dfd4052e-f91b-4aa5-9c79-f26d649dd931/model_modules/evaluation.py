from teradataml import DataFrame, create_context
from teradatasqlalchemy.types import INTEGER, VARCHAR, CLOB
from sklearn import metrics
from collections import OrderedDict
from aoa.sto.util import save_metadata, save_evaluation_metrics, check_sto_version
from .util import get_joined_models_df

import os
import numpy as np
import json
import base64
import dill


def evaluate(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]
    model_artefacts_table = "aoa_sto_models"

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    # validate that the python versions match between client and server
    check_sto_version()

    # Get the evaluation dataset. Note that we must join this dataset with the model artefacts we want to use in
    # evaluation. To do this, we join the model artefact table to the 1st row of the data table.
    # The util method does this for us.
    df = get_joined_models_df(data_table=data_conf["table"],
                              model_artefacts_table=model_artefacts_table,
                              model_version=model_version)

    # perform simple feature engineering example using map_row
    def transform_row(row):
        row["X1"] = row["X1"] + row["X1"] * 2.0
        return row

    df = df.map_row(lambda row: transform_row(row))

    # define evaluation logic/function we want to execute on each data partition.
    def eval_partition(partition):
        rows = partition.read()
        if rows is None or len(rows) == 0:
            return None

        # the model artefact is available on the 1st row only (see how we joined in the dataframe query)
        model_artefact = rows.loc[rows['n_row'] == 1, 'model_artefact'].iloc[0]
        model = dill.loads(base64.b64decode(model_artefact))

        X_test = rows[model.features]
        y_test = rows[["Y1"]]

        y_pred = model.predict(X_test)

        # record whatever partition level information you want like rows, data stats, metrics, explainability, etc
        partition_metadata = json.dumps({
            "num_rows": rows.shape[0],
            "metrics": {
                "MAE": "{:.2f}".format(metrics.mean_absolute_error(y_test, y_pred)),
                "MSE": "{:.2f}".format(metrics.mean_squared_error(y_test, y_pred)),
                "R2": "{:.2f}".format(metrics.r2_score(y_test, y_pred))
            }
        })

        # now return a single row for this partition with the evaluation results
        # (schema/order must match returns argument in map_partition)
        return np.array([[rows.partition_ID.iloc[0],
                          rows.shape[0],
                          partition_metadata]])

    print("Starting evaluation...")

    eval_df = df.map_partition(lambda partition: eval_partition(partition),
                               data_partition_column="partition_ID",
                               returns=OrderedDict(
                                   [('partition_id', VARCHAR(255)),
                                    ('num_rows', INTEGER()),
                                    ('partition_metadata', CLOB())]))

    # persist to temporary table for computing global metrics
    eval_df.to_sql("sto_eval_results", temporary=True)
    eval_df = DataFrame("sto_eval_results")

    save_metadata(eval_df)
    save_evaluation_metrics(eval_df, ["MAE", "MSE", "R2"])

    print("Finished evaluation")
