from teradataml import create_context
from tdextensions.distributed import DistDataFrame, DistMode
from sklearn import metrics
from .util import save_metadata

import os
import numpy as np
import json
import base64
import dill


def score(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]

    create_context(
        host=os.environ["AOA_CONN_HOST"],
        username=os.environ["AOA_CONN_USERNAME"],
        password=os.environ["AOA_CONN_PASSWORD"])

    def score_partition(partition):
        model_artefact = partition.loc[partition['n_row'] == 1, 'model_artefact'].iloc[0]
        model = dill.loads(base64.b64decode(model_artefact))

        X_test = partition[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

        y_pred = model.predict(X_test)
        partition_id = partition.species.iloc[0]

        return np.array([[partition_id, y_pred]])

    # we join the model artefact to the 1st row of the data table so we can load it in the partition
    query = """
    SELECT d.*, CASE WHEN n_row=1 THEN m.model_artefact ELSE null END AS model_artefact 
        FROM (SELECT x.*, row_number() OVER (PARTITION BY x.species ORDER BY x.species) AS n_row FROM iris_train x) AS d
        LEFT JOIN aoa_sto_models m
        ON d.species = m.partition_id
        WHERE model_artefact = '{model_version}'
    """.format(model_version=model_version)

    df = DistDataFrame(query, dist_mode=DistMode.STO, sto_id="my_model_score")
    scored_df = df.map_partition(lambda partition: score_partition(partition),
                                 partition_by="species",
                                 returns=[["partition_id", "VARCHAR(255)"],
                                          ["prediction", "VARCHAR(255)"]])

    scored_df.to_sql("my_predictions_table", if_exists="append")


def evaluate(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]

    create_context(
        host=os.environ["AOA_CONN_HOST"],
        username=os.environ["AOA_CONN_USERNAME"],
        password=os.environ["AOA_CONN_PASSWORD"])

    def eval_partition(partition):
        model_artefact = partition.loc[partition['n_row'] == 1, 'model_artefact'].iloc[0]
        model = dill.loads(base64.b64decode(model_artefact))

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

    # we join the model artefact to the 1st row of the data table so we can load it in the partition
    query = """
    SELECT d.*, CASE WHEN n_row=1 THEN m.model_artefact ELSE null END AS model_artefact 
        FROM (SELECT x.*, ROW_NUMBER() OVER (PARTITION BY x.species ORDER BY x.species) AS n_row FROM iris_train x) AS d
        LEFT JOIN aoa_sto_models m
        ON d.species = m.partition_id
        WHERE model_artefact = '{model_version}'
    """.format(model_version=model_version)

    df = DistDataFrame(query, dist_mode=DistMode.STO, sto_id="my_model_eval")
    eval_df = df.map_partition(lambda partition: eval_partition(partition),
                               partition_by="species",
                               returns=[["partition_id", "VARCHAR(255)"],
                                        ["num_rows", "BIGINT"],
                                        ["partition_metadata", "CLOB"]])

    metadata_df = eval_df.select(["partition_id", "partition_metadata", "num_rows"]).to_pandas()
    save_metadata(metadata_df, save_evaluation_metrics=True)
