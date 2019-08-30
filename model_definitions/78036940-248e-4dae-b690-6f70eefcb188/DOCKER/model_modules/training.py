import os

from teradataml import create_context
from teradataml.context.context import get_connection
from .util import execute_sql_script


def train(data_conf, model_conf, **kwargs):
    create_context(host=data_conf["hostname"],
                   username=os.environ["TD_USERNAME"],
                   password=os.environ["TD_PASSWORD"])

    print("Starting training...")
    
    sql_file = os.path.dirname(os.path.realpath(__file__)) +"/training.sql"
    jinja_ctx = {"data_conf": data_conf, "model_conf": model_conf}
    
    execute_sql_script(get_connection(), sql_file, jinja_ctx)
    
    print("Finished training")

    # save model 
    #get_connection().execute("INSERT INTO {} SELECT {}, T.* FROM {} T"
    #                         .format(data_conf["model_table"], '1', "csi_telco_churn_model"))

    print("Saved trained model")
