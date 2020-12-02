from teradataml import create_context
from tdextensions.distributed import DistDataFrame, DistMode
from sklearn.ensemble import RandomForestClassifier

import os
import numpy as np
import json
import base64
import dill


def save_metadata(model_df):
    # convert stats to dict and save to partitions.json
    metadata_df = model_df.select(["partition_id", "partition_metadata", "num_rows"]).to_pandas()
    metadata_dict = {r["partition_id"]: json.loads(r["partition_metadata"]) for r in
                     metadata_df.to_dict(orient='records')}

    with open("artifacts/output/partitions.json", 'w+') as f:
        json.dump(metadata_dict, f, indent=2)

    data_metadata = {
        "num_rows": int(metadata_df["num_rows"].sum())
    }

    with open("artifacts/output/data_stats.json", 'w+') as f:
        json.dump(data_metadata, f, indent=2)

    print("Finished saving artefacts")


def train(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]
    hyperparams = model_conf["hyperParameters"]

    create_context(host="host.docker.internal", username=os.environ['TD_USERNAME'], password=os.environ['TD_PASSWORD'])

    # hack until cli cleans this table up automatically or we allow to override on run
    from teradataml.context.context import get_connection
    get_connection().execute("DELETE FROM aoa_sto_models WHERE model_version='{}'".format(model_version))

    def train_partition(partition, model_version):
        X = partition[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y = partition[['species']]

        clf = RandomForestClassifier()
        clf.fit(X, y.values.ravel())

        partition_id = partition.species.iloc[0]
        artefact = base64.b64encode(dill.dumps(clf))

        # partition metadata allows you to record whatever partition level information you want from data statistics,
        # to hyper parameters used to model explainability
        partition_metadata = json.dumps({
            "num_rows": partition.shape[0],
            # "data_statistics": json.loads(partition.describe().to_json())
            # "explainability": shap.....
            "hyper_parameters": hyperparams
        })

        return np.array([[partition_id, model_version, partition.shape[0], partition_metadata, artefact]])

    print("Starting training...")

    df = DistDataFrame("iris_train", dist_mode=DistMode.STO, sto_id="my_model")
    model_df = df.map_partition(lambda partition: train_partition(partition, model_version),
                                partition_by="species",
                                returns=[["partition_id", "VARCHAR(255)"],
                                         ["model_version", "VARCHAR(255)"],
                                         ["num_rows", "BIGINT"],
                                         ["partition_metadata", "CLOB"],
                                         ["model_artefact", "CLOB"]])

    # append to models table
    model_df.to_sql("aoa_sto_models", if_exists="append")

    print("Finished training")

    save_metadata(model_df)
