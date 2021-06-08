from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from tdextensions.distributed import DistDataFrame, DistMode
from sklearn import metrics
from aoa.sto.util import save_metadata, save_evaluation_metrics

import os
import numpy as np
import json
import base64
import dill


def evaluate(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    def eval_partition(partition):
        model_artefact = partition.loc[partition['n_row'] == 1, 'model_artefact'].iloc[0]
        model = dill.loads(base64.b64decode(model_artefact))

        X_test = partition[model.features]
        y_test = partition[['Y1']]

        y_pred = model.predict(X_test)

        partition_id = partition.partition_ID.iloc[0]

        # record whatever partition level information you want like rows, data stats, metrics, explainability, etc
        partition_metadata = json.dumps({
            "num_rows": partition.shape[0],
            "metrics": {
                "MAE": "{:.2f}".format(metrics.mean_absolute_error(y_test, y_pred)),
                "MSE": "{:.2f}".format(metrics.mean_squared_error(y_test, y_pred)),
                "R2": "{:.2f}".format(metrics.r2_score(y_test, y_pred))
            }
        })

        return np.array([[partition_id, partition.shape[0], partition_metadata]])

    # we join the model artefact to the 1st row of the data table so we can load it in the partition
    query = f"""
    SELECT d.*, CASE WHEN n_row=1 THEN m.model_artefact ELSE null END AS model_artefact 
        FROM (SELECT x.*, ROW_NUMBER() OVER (PARTITION BY x.partition_id ORDER BY x.partition_id) AS n_row FROM {data_conf["table"]} x) AS d
        LEFT JOIN aoa_sto_models m
        ON d.partition_id = m.partition_id
        WHERE m.model_version = '{model_version}'
    """

    df = DistDataFrame(query=query, dist_mode=DistMode.STO, sto_id="model_eval")
    eval_df = df.map_partition(lambda partition: eval_partition(partition),
                               partition_by="partition_id",
                               returns=[["partition_id", "VARCHAR(255)"],
                                        ["num_rows", "BIGINT"],
                                        ["partition_metadata", "CLOB"]])

    # materialize as we reuse result
    eval_df = DataFrame(eval_df._table_name, materialize=True)

    save_metadata(eval_df)
    save_evaluation_metrics(eval_df, ["MAE", "MSE", "R2"])

    print("Finished evaluation")
