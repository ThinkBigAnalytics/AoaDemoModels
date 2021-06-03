from teradataml import create_context
from tdextensions.distributed import DistDataFrame, DistMode

import os
import base64
import dill


def score(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    def score_partition(partition):
        model_artefact = partition.loc[partition['n_row'] == 1, 'model_artefact'].iloc[0]
        model = dill.loads(base64.b64decode(model_artefact))

        X = partition[model.features]

        return model.predict(X)

    # we join the model artefact to the 1st row of the data table so we can load it in the partition
    query = f"""
    SELECT d.*, CASE WHEN n_row=1 THEN m.model_artefact ELSE null END AS model_artefact 
        FROM (SELECT x.*, ROW_NUMBER() OVER (PARTITION BY x.partition_id ORDER BY x.partition_id) AS n_row FROM {data_conf["table"]} x) AS d
        LEFT JOIN aoa_sto_models m
        ON d.partition_id = m.partition_id
        WHERE m.model_version = '{model_version}'
    """

    df = DistDataFrame(query=query, dist_mode=DistMode.STO, sto_id="my_model_score")
    scored_df = df.map_partition(lambda partition: score_partition(partition),
                                 partition_by="partition_id",
                                 returns=[["prediction", "VARCHAR(255)"]])

    scored_df.to_sql(data_conf["predictions"], if_exists="append")
