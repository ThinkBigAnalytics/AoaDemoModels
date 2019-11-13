import os

from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.analytics.mle import XGBoost
from teradataml.options.display import display

display.print_sqlmr_query = True


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    create_context(host=data_conf["hostname"],
                   username=os.environ["TD_USERNAME"],
                   password=os.environ["TD_PASSWORD"])

    dataset = DataFrame(data_conf['data_table'])

    print("Starting training...")

    # fit model to training data
    formula = "hasdiabetes ~ numtimesprg + plglcconc + bloodp + skinthick + twohourserins + bmi + dipedfunc + age"
    xgb = XGBoost(formula=formula,
                  data=dataset,
                  id_column='idx',
                  reg_lambda=float(hyperparams["reg_lambda"]),
                  shrinkage_factor=float(hyperparams["shrinkage_factor"]),
                  iter_num=10,
                  min_node_size=1,
                  max_depth=int(hyperparams["max_depth"]))

    # forces creation of model
    print(xgb.model_table)

    print("Finished training")

    xgb.model_table.to_sql(table_name=kwargs.get("model_table"))

    print("Saved model to table {}".format("model_table"))