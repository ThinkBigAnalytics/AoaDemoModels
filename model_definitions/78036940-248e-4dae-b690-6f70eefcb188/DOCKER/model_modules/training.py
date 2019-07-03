import os

from teradataml import create_context
from teradataml.context.context import get_connection
from .util import execute_sql_script


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    create_context(host=data_conf["hostname"],
                   username=os.environ["TD_USERNAME"],
                   password=os.environ["TD_PASSWORD"])

    print("Starting training...")

    execute_sql_script(get_connection(), "training.sql")
    
    print("Finished training")

    # save model 
    #get_connection().execute("INSERT INTO {} SELECT {}, T.* FROM {} T"
    #                         .format(data_conf["model_table"], '1', "csi_telco_churn_model"))

    print("Saved trained model")
