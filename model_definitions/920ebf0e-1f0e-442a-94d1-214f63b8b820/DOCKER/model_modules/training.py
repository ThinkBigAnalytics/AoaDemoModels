import os

from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.context.context import get_connection
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
    formula = "HasDiabetes ~ NumTimesPrg + PlGlcConc + BloodP + SkinThick + TwoHourSerIns + BMI + DiPedFunc + Age"
    xgb = XGBoost(formula=formula,
                  data=dataset,
                  id_column='idx',
                  reg_lambda=float(hyperparams["reg_lambda"]),
                  shrinkage_factor=hyperparams["shrinkage_factor"],
                  iter_num=10,
                  min_node_size=1,
                  max_depth=hyperparams["max_depth"])

    print("Finished training")

    # export model artefacts
    # xgb.model_table.to_sql(table_name=data_conf["model_table"], if_exists="replace")
    get_connection().execute("INSERT INTO aoa_models SELECT {}, T.* FROM {} T"
                             .format('1', xgb.model_table._table_name))

    # model = xgb.model_table.to_pandas()
    # model.to_hdf("models/model.h5", key="model", mode="w")

    print("Saved trained model")
