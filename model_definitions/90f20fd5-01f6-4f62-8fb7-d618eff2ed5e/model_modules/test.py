from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.dataframe.copy_to import copy_to_sql
from aoa.stats import stats

import joblib
import pandas as pd
import json
import logging
import sys
import jinja2
import importlib
import os

with open("trained_model.json") as f:
    trained_model = json.load(f)

template_env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath="."))
template = template_env.get_template("scheduler/dataset_template.json")

data_conf = json.loads(template.render(dict(os.environ)))

create_context(host=os.environ["AOA_CONN_HOST"],
               username=os.environ["AOA_CONN_USERNAME"],
               password=os.environ["AOA_CONN_PASSWORD"],
               database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

features_tdf = DataFrame(data_conf["table"]).sample(5)
predictions_tdf = DataFrame(data_conf["predictions"]).sample(5)
trained_model = {
    "model_version": trained_model["id"],
    "model_id": trained_model["modelId"],
    "project_id": trained_model["projectId"],
}
stats.record_scoring_stats(features_tdf, predictions_tdf, trained_model)
