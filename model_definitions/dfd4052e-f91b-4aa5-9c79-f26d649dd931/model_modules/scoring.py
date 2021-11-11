from teradataml import DataFrame, create_context
from teradatasqlalchemy.types import VARCHAR
from collections import OrderedDict
from aoa.sto.util import check_sto_version

import os
import base64
import dill


def score(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]
    model_table = "aoa_sto_models"

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    check_sto_version()

    def score_partition(partition):
        rows = partition.read()
        if rows is None:
            return None

        model_artefact = rows.loc[rows['n_row'] == 1, 'model_artefact'].iloc[0]
        model = dill.loads(base64.b64decode(model_artefact))

        out_df = rows[["ID"]]
        out_df["predictions"] = model.predict(rows[model.features])

        return out_df

    print("Starting scoring...")

    # we join the model artefact to the 1st row of the data table so we can load it in the partition
    query = f"""
    SELECT d.*, CASE WHEN n_row=1 THEN m.model_artefact ELSE null END AS model_artefact 
        FROM (SELECT x.*, ROW_NUMBER() OVER (PARTITION BY x.partition_id ORDER BY x.partition_id) AS n_row FROM {data_conf["table"]} x) AS d
        LEFT JOIN {model_table} m
        ON d.partition_id = m.partition_id
        WHERE m.model_version = '{model_version}'
    """

    df = DataFrame(query=query)
    scored_df = df.map_partition(lambda partition: score_partition(partition),
                                 data_partition_column="partition_ID",
                                 returns=OrderedDict(
                                     [('partition_id', VARCHAR(255)),
                                      ('prediction', VARCHAR(255))]))

    scored_df.to_sql(data_conf["predictions"], if_exists="append")

    print(scored_df.head(2))

    print("Finished scoring...")
