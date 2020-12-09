from teradataml import create_context
from tdextensions.distributed import DistDataFrame, DistMode
from teradataml.dataframe.dataframe import DataFrame
from sklearn import metrics
from .util import save_metadata

import os
import numpy as np
import json
import base64
import dill


def score(data_conf, model_conf, **kwargs):
    pass


def evaluate(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]

    create_context(
        host=os.environ["AOA_CONN_HOST"],
        username=os.environ["AOA_CONN_USERNAME"],
        password=os.environ["AOA_CONN_PASSWORD"])

    def eval_partition(partition):
        model = dill.loads(base64.b64decode(partition["model_artefact"].iloc[0]))

        X_test = partition[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y_test = partition[['species']]

        y_pred = model.predict(X_test)

        partition_id = partition.species.iloc[0]

        # record whatever partition level information you want like rows, data stats, metrics, explainability, etc
        partition_metadata = json.dumps({
            "num_rows": partition.shape[0],
            "metrics": {
                "Accuracy": "{:.2f}".format(metrics.accuracy_score(y_test, y_pred)),
                "Recall": "{:.2f}".format(metrics.recall_score(y_test, y_pred)),
                "Precision": "{:.2f}".format(metrics.precision_score(y_test, y_pred)),
                "f1-score": "{:.2f}".format(metrics.f1_score(y_test, y_pred))
            }
        })

        return np.array([[partition_id, partition.shape[0], partition_metadata]])

    df = DataFrame.from_query("""
    SELECT i.*, m.model_artefact FROM iris_train i 
        LEFT JOIN aoa_sto_models m ON i.species = m.partition_id 
        WHERE model_version='{}'
    """.format(model_version))

    df = DistDataFrame(df._table_name, dist_mode=DistMode.STO, sto_id="my_model")
    eval_df = df.map_partition(lambda partition: eval_partition(partition),
                               partition_by="species",
                               returns=[["partition_id", "VARCHAR(255)"],
                                        ["num_rows", "BIGINT"],
                                        ["partition_metadata", "CLOB"]])

    metadata_df = eval_df.select(["partition_id", "partition_metadata", "num_rows"]).to_pandas()
    save_metadata(metadata_df, save_evaluation_metrics=True)
