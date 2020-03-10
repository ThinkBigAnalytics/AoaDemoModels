import os
import json

from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.analytics.mle import XGBoostPredict, ConfusionMatrix
from teradataml.options.display import display

display.print_sqlmr_query = True


def score(data_conf, model_conf, **kwargs):
    create_context(host=data_conf["hostname"],
                   username=os.environ["TD_USERNAME"],
                   password=os.environ["TD_PASSWORD"],
                   logmech=os.getenv("TD_LOGMECH", "TDNEGO"))

    model = DataFrame(kwargs.get("model_table"))

    dataset = DataFrame(data_conf['data_table'])

    print("Starting Scoring...")

    predicted = XGBoostPredict(object=model,
                               newdata=dataset,
                               id_column='idx',
                               object_order_column=['tree_id', 'iter', 'class_num'],
                               terms=["hasdiabetes"])

    # lazy evaluation so trigger it. export evaluation results (temporary=True causes issue)
    predicted.result.to_sql(data_conf["predictions_table"], if_exists="replace")

    print("Finished Scoring")


def evaluate(data_conf, model_conf, **kwargs):

    score(data_conf, model_conf, **kwargs)

    print("Starting Comparison")

    cm = ConfusionMatrix(data=DataFrame(data_conf["predictions_table"]),
                         reference='hasdiabetes',
                         prediction='prediction')

    print("Confusion Matrix Stats: {}".format(cm.stattable))

    with open("models/metrics.json", "w+") as f:
        metrics = cm.stattable.to_pandas()
        metrics = dict(zip(metrics.key, metrics.value))

        json.dump(metrics, f)

    print("Finished Comparison")
