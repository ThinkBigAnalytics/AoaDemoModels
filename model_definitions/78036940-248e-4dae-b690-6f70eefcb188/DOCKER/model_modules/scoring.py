import os
import json

from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.context.context import get_connection
from .util import execute_sql_script


def evaluate(data_conf, model_conf, **kwargs):
    create_context(host=data_conf["hostname"],
                   username=os.environ["TD_USERNAME"],
                   password=os.environ["TD_PASSWORD"])
    
    print("Starting evaluation...")

    execute_sql_script(get_connection(), "scoring.sql")
    
    print("Finished evaluation")
    
    stats = DataFrame("telco_stat_output").to_pandas()
    metrics = dict(zip(stats.key, stats.value))
    
    with open("models/evaluation.json", "w+") as f:
        json.dump(metrics, f)

    