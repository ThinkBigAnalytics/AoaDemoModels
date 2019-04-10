import os
import json

from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.analytics.mle import XGBoostPredict
from teradataml.options.display import display
# from teradataml.options.configure import configure

# configure.default_varchar_size = 100*1024
display.print_sqlmr_query = True


def evaluate(data_conf, model_conf, **kwargs):
    create_context(host=data_conf["hostname"],
                   username=os.environ["USERNAME"],
                   password=os.environ["PASSWORD"])

    # minor bug in python lib that doesn't allow creating large columns as CLOB etc.
    # model = pd.read_hdf('models/model.h5', 'model')
    # copy_to_sql(df=model, table_name="pima_model", index=True, index_label="idx", if_exists="replace")
    model = DataFrame(data_conf["model_table"])

    dataset = DataFrame(data_conf['data_table'])
    dataset = dataset[dataset['idx'] >= 600]

    print("Starting Evaluation...")

    predicted = XGBoostPredict(object=model,
                               newdata=dataset,
                               id_column='idx',
                               object_order_column=['tree_id', 'iter', 'class_num'],
                               terms=["HasDiabetes"])

    print("Finished Evaluation")

    # export evaluation results (temporary=True causes issue)
    predicted.result.to_sql("pima_predictions", if_exists="replace")

    summary = DataFrame.from_query(
        "SELECT count(*) as match FROM pima_predictions WHERE HasDiabetes = CAST(prediction AS BIGINT)")
    matches = summary.to_pandas()["match"].values[0]
    accuracy = matches / dataset.shape[0]
    print("Accuracy: {}".format(accuracy))

    with open("models/evaluation.json", "w+") as f:
        json.dump({'accuracy': (accuracy * 100.0)}, f)

    print("Saved evaluation results")
