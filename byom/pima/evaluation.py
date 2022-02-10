from sklearn import metrics
from teradataml import (
    get_context,
    DataFrame,
    PMMLPredict,
    configure
)
from aoa.stats import stats
from aoa.util import (
    aoa_create_context,
    store_byom_tmp
)

import os
import json

configure.byom_install_location = os.environ.get("AOA_BYOM_INSTALL_DB", "MLDB")


def plot_confusion_matrix(cf):
    import itertools
    import matplotlib.pyplot as plt
    plt.imshow(cf, cmap=plt.cm.Blues, interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['0', '1'])
    plt.yticks([0, 1], ['0', '1'])

    thresh = cf.max() / 2.
    for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
        plt.text(j, i, format(cf[i, j], 'd'), horizontalalignment='center',
                 color='white' if cf[i, j] > thresh else 'black')

    fig = plt.gcf()
    fig.savefig('artifacts/output/confusion_matrix', dpi=500)
    plt.clf()


def evaluate(data_conf, model_conf, model_version, **kwargs):
    aoa_create_context()

    with open("artifacts/input/model.pmml", "rb") as f:
        model_bytes = f.read()

    model = store_byom_tmp(get_context(), "ivsm_models_tmp", model_version, model_bytes)

    pmml = PMMLPredict(
        modeldata=model,
        newdata=DataFrame(data_conf["table"]),
        accumulate=["PatientId", "HasDiabetes"])

    pmml.result.to_sql(table_name="predictions_tmp", if_exists="replace", temporary=True)

    metrics_df = DataFrame.from_query("""
    SELECT 
        HasDiabetes as y_test, 
        CAST(CAST(json_report AS JSON).JSONExtractValue('$.predicted_HasDiabetes') AS INT) as y_pred
        FROM predictions_tmp
    """)
    metrics_df = metrics_df.to_pandas()

    y_pred = metrics_df[["y_pred"]]
    y_test = metrics_df[["y_test"]]

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
    }

    with open("artifacts/output/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    # create confusion matrix plot
    cf = metrics.confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(cf)

    # calculate stats if training stats exist
    if os.path.exists("artifacts/input/data_stats.json"):
        stats.record_evaluation_stats(DataFrame(data_conf["table"]), DataFrame("predictions_tmp"))
