import os

from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.analytics.mle import XGBoost
from teradataml.options.display import display

display.print_sqlmr_query = True


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    create_context(host=data_conf["hostname"],
                   username=os.environ["USERNAME"],
                   password=os.environ["PASSWORD"])

    dataset = DataFrame(data_conf['data_table'])
    dataset = dataset[dataset['idx'] < 600]

    print("Starting training...")

    # fit model to training data
    formula = "HasDiabetes ~ NumTimesPrg + PlGlcConc + BloodP + SkinThick + TwoHourSerIns + BMI + DiPedFunc + Age"
    xgb = XGBoost(formula=formula,
                  data=dataset,
                  id_column='idx',
                  reg_lambda=hyperparams["reg_lambda"],
                  shrinkage_factor=hyperparams["shrinkage_factor"],
                  iter_num=10,
                  min_node_size=1,
                  max_depth=hyperparams["max_depth"])

    print("Finished training")

    # export model artefacts
    xgb.model_table.to_sql(table_name="pima_model", if_exists="replace")

    model = xgb.model_table.to_pandas()
    model.to_hdf("models/model.h5", key="model", mode="w")

    print("Saved trained model")
