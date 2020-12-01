from teradataml import create_context
from tdextensions.distributed import DistDataFrame, DistMode
from teradataml.dataframe.dataframe import DataFrame

import base64
import dill
import os
import numpy as np
import json


def score(data_conf, model_conf, **kwargs):
    """Python score method called by AOA framework batch mode

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """


def evaluate(data_conf, model_conf, **kwargs):
    """Python evaluate method called by AOA framework

   evaluate
    - normal stuff
just mock the eval to start and return metrics !!
When implementing just join models to start
    - evaluate per partition and collect results back. Produce aggregate /histogram charts / distribution
 - save individual metrics in partition_metrics.json

    """

    model_version = kwargs["model_version"]

    models_df = DataFrame.from_query("SELECT * FROM aoa_sto_models WHERE model_version='{}'".format(model_version))
    df = DataFrame("iris_train")
    df = df.join(models_df, on=[df.species == models_df.partition_id], how="left")

    def eval_partition(partition):
        return "1"

    df = DistDataFrame(df._table_name, dist_mode=DistMode.STO, sto_id="my_model")
    eval_df = df.map_partition(lambda partition: eval_partition(partition),
                               partition_by="species",
                               returns=[["partition_id", "VARCHAR(255)"]])

    # dump results as json file evaluation.json to models/ folder
    print("Evaluation complete...")


# Uncomment this code if you want to deploy your model as a Web Service (Real-time / Interactive usage)
# class ModelScorer(object):
#    def __init__(self, config=None):
#        self.model = joblib.load('models/iris_knn.joblib')
#
#    def predict(self, data):
#        return self.model.predict([data])
#