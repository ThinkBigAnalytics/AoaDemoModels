from sklearn import metrics
from teradataml import get_context, copy_to_sql, DataFrame, get_connection
from aoa.stats import stats
from aoa.util import aoa_create_context

import json
import itertools
import pandas as pd
import matplotlib.pyplot as plt


def evaluate(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]
    model_id = kwargs["model_id"]

    aoa_create_context()

    with open("artifacts/input/model.pmml", "rb") as f:
        model_bytes = f.read()

    # we don't want to insert this into the models that can be used yet so add to temporary table and use there
    get_context().execute("""
    CREATE VOLATILE TABLE ivsm_models_tmp(
        model_version VARCHAR(255),
        model_id VARCHAR(255),
        model BLOB(2097088000)
    ) ON COMMIT PRESERVE ROWS;
    """)
    get_context().execute(f"INSERT INTO ivsm_models_tmp(model_version, model_id, model) "
                   "values(?,?,?)",
                   (model_version, model_id, model_bytes))

    scores_df = pd.read_sql(f"""
    SELECT PatientId, HasDiabetes as y_test, CAST(CAST(score_result AS JSON).JSONExtractValue('$.predicted_HasDiabetes') AS INT) as y_pred FROM IVSM.IVSM_SCORE(
                ON (SELECT * FROM {data_conf["table"]}) AS DataTable
                ON (SELECT model_id, model FROM ivsm_models_tmp WHERE model_version = '{model_version}') AS ModelTable DIMENSION
                USING
                    ModelID('{model_id}')
                    ColumnsToPreserve('PatientId', 'HasDiabetes')
                    ModelType('PMML')
            ) sc;
    """, get_connection())

    y_pred = scores_df[["y_pred"]]
    y_test = scores_df[["y_test"]]

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

    plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['0','1'])
    plt.yticks([0, 1], ['0','1'])

    thresh = cf.max() / 2.
    for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
        plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')

    fig = plt.gcf()
    fig.savefig('artifacts/output/confusion_matrix', dpi=500)
    plt.clf()

    predictions_table = "{}_tmp".format(data_conf["predictions"]).lower()
    predictions_df = scores_df[["y_pred"]].rename({'y_pred': 'HasDiabetes'}, axis=1)
    copy_to_sql(df=predictions_df, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

    stats.record_evaluation_stats(DataFrame(data_conf["table"]), DataFrame(predictions_table))
