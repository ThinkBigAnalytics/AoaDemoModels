from teradataml import DataFrame, create_context
from teradatasqlalchemy.types import INTEGER, VARCHAR, CLOB
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
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
    model_table = "aoa_sto_models"

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    check_sto_version()
    cleanup_cli(model_version, model_table)

    def train_partition(partition, model_version, hyperparams):
        rows = partition.read()
        if rows is None:
            return None

        numeric_features = ["X"+str(i) for i in range(1,10)]
        for i in numeric_features:
            rows[i] = rows[i].astype("float")

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("pca", PCA(0.95))
        ])

        categorical_features = ["flag"]
        for i in categorical_features:
            rows[i] = rows[i].astype("category")

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))])

        preprocessor = ColumnTransformer(transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)])

        features = numeric_features + categorical_features
        pipeline = Pipeline([("preprocessor", preprocessor),
                             ("rf", RandomForestRegressor(max_depth=hyperparams["max_depth"]))])
        pipeline.fit(rows[features], rows[['Y1']])
        pipeline.features = features

        partition_id = rows.partition_ID.iloc[0]
        artefact = base64.b64encode(dill.dumps(pipeline))

        # record whatever partition level information you want like rows, data stats, explainability, etc
        partition_metadata = json.dumps({
            "num_rows": rows.shape[0],
            "hyper_parameters": hyperparams
        })

        return np.array([[partition_id, model_version, rows.shape[0], partition_metadata, artefact]])

    print("Starting training...")

    query = "SELECT * FROM {table} WHERE fold_ID='train'".format(table=data_conf["table"])
    df = DataFrame(query=query)
    model_df = df.map_partition(lambda partition: train_partition(partition, model_version, hyperparams),
                                data_partition_column="partition_ID",
                                returns=OrderedDict(
                                    [('partition_id', VARCHAR(255)),
                                     ('model_version', VARCHAR(255)),
                                     ('num_rows', INTEGER()),
                                     ('partition_metadata', CLOB()),
                                     ('model_artefact', CLOB())]))

    # persist to models table
    model_df.to_sql(model_table, if_exists="append")
    model_df = DataFrame(query=f"SELECT * FROM {model_table} WHERE model_version='{model_version}'")

    save_metadata(model_df)

    print("Finished training")

    with open("artifacts/output/sto_versions.json", "w+") as f:
        json.dump(collect_sto_versions(), f)
