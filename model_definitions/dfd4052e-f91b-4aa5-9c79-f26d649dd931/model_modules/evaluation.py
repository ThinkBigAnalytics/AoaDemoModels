from teradataml import DataFrame, create_context
from teradatasqlalchemy.types import INTEGER, VARCHAR, CLOB
from sklearn import metrics
from collections import OrderedDict
from aoa.sto.util import save_metadata, save_evaluation_metrics, check_sto_version

import os
import numpy as np
import json
import base64
import dill


def evaluate(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]
    model_table = "aoa_sto_models"

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    check_sto_version()

    def eval_partition(partition):
        rows = partition.read()
        if rows is None:
            return None

        model_artefact = rows.loc[rows['n_row'] == 1, 'model_artefact'].iloc[0]
        model = dill.loads(base64.b64decode(model_artefact))

        X_test = rows[model.features]
        y_test = rows[['Y1']]

        y_pred = model.predict(X_test)

        partition_id = rows.partition_ID.iloc[0]

        # record whatever partition level information you want like rows, data stats, metrics, explainability, etc
        partition_metadata = json.dumps({
            "num_rows": rows.shape[0],
            "metrics": {
                "MAE": "{:.2f}".format(metrics.mean_absolute_error(y_test, y_pred)),
                "MSE": "{:.2f}".format(metrics.mean_squared_error(y_test, y_pred)),
                "R2": "{:.2f}".format(metrics.r2_score(y_test, y_pred))
            }
        })

        return np.array([[partition_id, rows.shape[0], partition_metadata]])

    print("Starting evaluation...")

    # we join the model artefact to the 1st row of the data table so we can load it in the partition
    query = f"""
    SELECT d.*, CASE WHEN n_row=1 THEN m.model_artefact ELSE null END AS model_artefact 
        FROM (SELECT x.*, ROW_NUMBER() OVER (PARTITION BY x.partition_id ORDER BY x.partition_id) AS n_row FROM {data_conf["table"]} x) AS d
        LEFT JOIN {model_table} m
        ON d.partition_id = m.partition_id
        WHERE m.model_version = '{model_version}'
    """

    df = DataFrame(query=query)
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
