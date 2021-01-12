from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from tdextensions.distributed import DistDataFrame, DistMode
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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

    def train_partition(partition, model_version, hyperparams):
        numeric_features = ["X"+str(i) for i in range(1,10)]
        for i in numeric_features:
            partition[i] = partition[i].astype("float")

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("pca",PCA(0.95))
        ])

        categorical_features = ["flag"]
        for i in categorical_features:
            partition[i] = partition[i].astype("category")

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))])

        preprocessor = ColumnTransformer(transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)])

        features = numeric_features + categorical_features
        pipeline = Pipeline([("preprocessor", preprocessor),
                             ("rf", RandomForestRegressor(max_depth=hyperparams["max_depth"]))])
        pipeline.fit(partition[features], partition[['Y1']])
        pipeline.features = features

        partition_id = partition.partition_ID.iloc[0]
        artefact = base64.b64encode(dill.dumps(pipeline))

        # record whatever partition level information you want like rows, data stats, explainability, etc
        partition_metadata = json.dumps({
            "num_rows": partition.shape[0],
            "hyper_parameters": hyperparams
        })

        return np.array([[partition_id, model_version, partition.shape[0], partition_metadata, artefact]])

    print("Starting training...")

    query = "SELECT * FROM {table} WHERE fold_ID='train'".format(table=data_conf["table"])
    df = DistDataFrame(query=query, dist_mode=DistMode.STO, sto_id="model_train")
    model_df = df.map_partition(lambda partition: train_partition(partition, model_version, hyperparams),
                                partition_by="partition_id",
                                returns=[["partition_id", "VARCHAR(255)"],
                                         ["model_version", "VARCHAR(255)"],
                                         ["num_rows", "BIGINT"],
                                         ["partition_metadata", "CLOB"],
                                         ["model_artefact", "CLOB"]])
    # materialize as we reuse result
    model_df = DataFrame(model_df._table_name, materialize=True)

    # append to models table
    model_df.to_sql("aoa_sto_models", if_exists="append")

    save_metadata(model_df)

    print("Finished training")
