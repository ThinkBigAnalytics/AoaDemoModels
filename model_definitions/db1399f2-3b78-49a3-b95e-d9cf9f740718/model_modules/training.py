from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.context.context import get_connection
from teradataml.analytics.mle import GLM
from teradataml.options.display import display
import os


display.print_sqlmr_query = True


def train(data_conf, model_conf, **kwargs):

    hyperparams = model_conf["hyperParameters"]

    create_context(host=data_conf["hostname"],
                   username=os.environ['TD_USERNAME'],
                   password=os.environ['TD_PASSWORD'])

    dataset = DataFrame(data_conf['data_table'])

    print("Starting training...")

    # fit model to training data
    formula = model_conf["formula"]

    glm = GLM(
        formula = formula, 
        family = model_conf["family"],
        linkfunction = model_conf["linkfunction"], 
        data = dataset, 
        
        threshold = float(hyperparams["threshold"]), 
        maxit = int(hyperparams["maxit"]), 
        step = bool(hyperparams["step"]),
        intercept = bool(hyperparams["intercept"])
        )

    
    model = glm.coefficients
    # export model artefacts
    model.to_sql(table_name=data_conf["model_table"], if_exists="replace")
    print("Finished training")
    print("Saved trained model")
