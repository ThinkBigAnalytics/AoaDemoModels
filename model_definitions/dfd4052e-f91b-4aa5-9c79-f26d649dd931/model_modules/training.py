from teradataml import create_context
from tdextensions.distributed import DistDataFrame, DistMode
from sklearn.ensemble import RandomForestClassifier
from .util import save_metadata, cleanup_cli

import os
import numpy as np
import json
import base64
import dill


def train(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]
    hyperparams = model_conf["hyperParameters"]

    create_context(
        host=os.environ["AOA_CONN_HOST"],
        username=os.environ["AOA_CONN_USERNAME"],
        password=os.environ["AOA_CONN_PASSWORD"])

    cleanup_cli(model_version)

    def train_partition(partition, kwargs, hyperparams):
        X = partition[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y = partition[['species']]

        model = RandomForestClassifier()
        model.fit(X, y.values.ravel())

        partition_id = partition.species.iloc[0]
        artefact = base64.b64encode(dill.dumps(model))

        # record whatever partition level information you want like rows, data stats, explainability, etc
        partition_metadata = json.dumps({
            "num_rows": partition.shape[0],
            # "data_statistics": json.loads(partition.describe().to_json())
            # "explainability": shap.....
            "hyper_parameters": hyperparams
        })

        return np.array([[partition_id, model_version, partition.shape[0], partition_metadata, artefact]])

    print("Starting training...")

    df = DistDataFrame("iris_train", dist_mode=DistMode.STO, sto_id="my_model_train")
    model_df = df.map_partition(lambda partition: train_partition(partition, model_version, hyperparams),
                                partition_by="species",
                                returns=[["partition_id", "VARCHAR(255)"],
                                         ["model_version", "VARCHAR(255)"],
                                         ["num_rows", "BIGINT"],
                                         ["partition_metadata", "CLOB"],
                                         ["model_artefact", "CLOB"]])

    # append to models table
    model_df.to_sql("aoa_sto_models", if_exists="append")

    print("Finished training")

    metadata_df = model_df.select(["partition_id", "partition_metadata", "num_rows"]).to_pandas()
    save_metadata(metadata_df)
