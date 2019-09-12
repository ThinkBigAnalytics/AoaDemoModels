from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.context.context import get_connection
from teradataml.analytics.mle import DecisionForest
from teradataml.options.display import display
import os


display.print_sqlmr_query = True


def train(data_conf, model_conf, **kwargs):

    create_context(host=data_conf["hostname"],
                   username=os.environ['TD_USERNAME'],
                   password=os.environ['TD_PASSWORD'])

    dataset = DataFrame(data_conf['data_table'])

    print("Starting training...")

    # fit model to training data
    formula = model_conf["formula"]

    hyperparams = model_conf["hyperParameters"]

    tree_type = model_conf["tree_type"]

    df = DecisionForest(formula=formula,
                        data=dataset,
                        tree_type = tree_type,

                        ntree=int(hyperparams["ntree"]),
                        nodesize=int(hyperparams["nodesize"]),
                        max_depth=int(hyperparams["max_depth"]),
                        mtry = int(hyperparams["mtry"])
                        )

    print("Finished training")
    model = df.predictive_model
    # export model artefacts
    model.to_sql(table_name=data_conf["model_table"], if_exists="replace")

    print("Saved trained model")
