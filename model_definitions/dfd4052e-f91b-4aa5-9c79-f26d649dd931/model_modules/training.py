from teradataml import create_context
from tdextensions.distributed import DistDataFrame, DistMode
from sklearn.ensemble import RandomForestClassifier

import base64
import dill
import os
import numpy as np
import json


def train(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]

    #data_conf["host"]
    create_context(host="host.docker.internal", username=os.environ['TD_USERNAME'], password=os.environ['TD_PASSWORD'])

    def train_partition(partition, model_version):
        X = partition[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y = partition[['species']]

        clf = RandomForestClassifier()
        clf.fit(X, y.values.ravel())

        partition_id = partition[['species']].iloc[0]

        # the following can be wrapped up in a library call to generate whatever stats we want
        artefact = base64.b64encode(dill.dumps(clf))
        stats = json.dumps({"rows": partition.shape[0]})

        return np.array([[partition_id, model_version, stats, artefact]])

    print("Starting training...")

    df = DistDataFrame("iris_train", dist_mode=DistMode.STO, sto_id="my_model")
    model_df = df.map_partition(lambda partition: train_partition(partition, model_version),
                                partition_by="species",
                                returns=[["partition_id", "VARCHAR(255)"],
                                         ["model_version", "VARCHAR(255)"],
                                         ["partition_stats", "CLOB"],
                                         ["model_artefact", "CLOB"]])

    # append to models table
    model_df.to_sql("aoa_sto_models", if_exists="append")

    print("Finished training")

    # convert stats to dict and save to partitions.json
    stats_df = model_df.select(["partition_id", "partition_stats"]).to_pandas()
    stats_dict = {r["partition_id"]: json.loads(r["partition_stats"]) for r in stats_df.to_dict(orient='records')}

    with open("artifacts/output/partitions.json", 'w+') as f:
        json.dump(stats_dict, f, indent=2)

    print("Finished saving artefacts")
