from teradataml import create_context
from teradatasqlalchemy.types import VARCHAR
from collections import OrderedDict
from aoa.sto.util import check_sto_version
from .util import get_joined_models_df

import os
import base64
import dill


def score(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]
    model_artefacts_table = "aoa_sto_models"

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    # validate that the python versions match between client and server
    check_sto_version()

    # Get the scoring dataset. Note that we must join this dataset with the model artefacts we want to use in
    # evaluation. To do this, we join the model artefact table to the 1st row of the data table.
    # The util method does this for us.
    df = get_joined_models_df(data_table=data_conf["table"],
                              model_artefacts_table=model_artefacts_table,
                              model_version=model_version)

    # perform simple feature engineering example using map_row
    def transform_row(row):
        row["X1"] = row["X1"] + row["X1"] * 2.0
        return row

    df = df.map_row(lambda row: transform_row(row))

    # define scoring logic/function we want to execute on each data partition.
    def score_partition(partition):
        rows = partition.read()
        if rows is None or len(rows) == 0:
            return None

        # the model artefact is available on the 1st row only (see how we joined in the dataframe query)
        model_artefact = rows.loc[rows['n_row'] == 1, 'model_artefact'].iloc[0]
        model = dill.loads(base64.b64decode(model_artefact))

        out_df = rows[["ID"]]
        out_df["predictions"] = model.predict(rows[model.features])

        # now return a single row for this partition with the prediction results
        # (schema/order must match returns argument in map_partition)
        return out_df

    print("Starting scoring...")

    scored_df = df.map_partition(lambda partition: score_partition(partition),
                                 data_partition_column="partition_ID",
                                 returns=OrderedDict(
                                     [('partition_id', VARCHAR(255)),
                                      ('prediction', VARCHAR(255))]))

    # append the predictions to the predictions table
    scored_df.to_sql(data_conf["predictions"], if_exists="append")

    print("Finished scoring...")
