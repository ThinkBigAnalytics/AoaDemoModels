from teradataml import DataFrame, create_context
from teradatasqlalchemy.types import INTEGER, VARCHAR, CLOB
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from aoa.sto.util import save_metadata, cleanup_cli, check_sto_version, collect_sto_versions
from collections import OrderedDict

import os
import numpy as np
import json
import base64
import dill


def train(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]
    hyperparams = model_conf["hyperParameters"]
    model_artefacts_table = "aoa_sto_models"

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    # validate that the python versions match between client and server
    check_sto_version()

    # required if executing multiple times via cli (model_version = 'cli' on every run).
    cleanup_cli(model_version)

    # select the training datast via the fold_id
    query = "SELECT * FROM {table} WHERE fold_id='train'".format(table=data_conf["table"])
    df = DataFrame(query=query)

    # perform simple feature engineering example using map_row
    def transform_row(row):
        row["X1"] = row["X1"] + row["X1"] * 2.0
        return row

    df = df.map_row(lambda row: transform_row(row))

    # define training logic/function we want to execute on each data partition.
    def train_partition(partition, model_version, hyperparams):
        rows = partition.read()
        if rows is None or len(rows) == 0:
            return None

        features = ["X1", "X2", "X3"]

        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", RobustScaler()),
                             ("rf", RandomForestRegressor(max_depth=hyperparams["max_depth"]))])
        pipeline.features = features

        pipeline.fit(rows[features], rows[["Y1"]])

        # record whatever partition level information you want like rows, data stats, explainability, etc
        partition_metadata = json.dumps({
            "num_rows": rows.shape[0],
            "hyper_parameters": hyperparams
        })

        # now return a single row for this partition with the model details and artefact
        # (schema/order must match returns argument in map_partition)
        return np.array([[rows.partition_ID.iloc[0],
                          model_version,
                          rows.shape[0],
                          partition_metadata,
                          base64.b64encode(dill.dumps(pipeline))]])

    print("Starting training...")

    model_df = df.map_partition(lambda partition: train_partition(partition, model_version, hyperparams),
                                data_partition_column="partition_ID",
                                returns=OrderedDict(
                                    [('partition_id', VARCHAR(255)),
                                     ('model_version', VARCHAR(255)),
                                     ('num_rows', INTEGER()),
                                     ('partition_metadata', CLOB()),
                                     ('model_artefact', CLOB())]))

    # persist to models table
    model_df.to_sql(model_artefacts_table, if_exists="append")
    model_df = DataFrame(query=f"SELECT * FROM {model_artefacts_table} WHERE model_version='{model_version}'")

    save_metadata(model_df)

    print("Finished training")

    with open("artifacts/output/sto_versions.json", "w+") as f:
        json.dump(collect_sto_versions(), f)

