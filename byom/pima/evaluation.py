from sklearn import metrics
from teradataml import (
    get_context,
    DataFrame,
    PMMLPredict,
    configure
)
from aoa import (
    record_evaluation_stats,
    aoa_create_context,
    store_byom_tmp,
    ModelContext
)

import os
import json

configure.byom_install_location = os.environ.get("AOA_BYOM_INSTALL_DB", "MLDB")


def plot_confusion_matrix(cf, img_filename):
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
    fig.savefig(img_filename, dpi=500)
    plt.clf()


def evaluate(context: ModelContext, **kwargs):
    aoa_create_context()

    
    # this evaluation.py can hanlde both onnx and pmml. usually, you would only need to support one but for 
    # demo purposes, we will show with both as we produce both onnx and pmml in this notebook.
    
    import glob
    for file_name in glob.glob(f"{context.artifact_input_path}/model.*"):
        model_type = file_name.split(".")[-1]
    
    with open(f"{context.artifact_input_path}/model.{model_type}", "rb") as f:
        model_bytes = f.read()
        
    model = store_byom_tmp(get_context(), "byom_models_tmp", context.model_version, model_bytes)

    target_name = context.dataset_info.target_names[0]

    if model_type.upper() == "ONNX":
        byom_target_sql = "CAST(CAST(json_report AS JSON).JSONExtractValue('$.output_label[0]') AS INT)"
        mldb = os.environ.get("AOA_BYOM_INSTALL_DB", "MLDB")

        query = f"""
            SELECT sc.{context.dataset_info.entity_key}, {target_name}, sc.json_report
                FROM {mldb}.ONNXPredict(
                    ON ({context.dataset_info.sql}) AS DataTable
                    ON (SELECT model_version as model_id, model FROM byom_models_tmp) AS ModelTable DIMENSION
                    USING
                        Accumulate('{context.dataset_info.entity_key}', '{target_name}')
            ) sc;
        """

        predictions_df = DataFrame.from_query(query)
        
    elif model_type.upper() == "PMML":
        byom_target_sql = "CAST(CAST(json_report AS JSON).JSONExtractValue('$.predicted_HasDiabetes') AS INT)"
        
        pmml = PMMLPredict(
            modeldata=model,
            newdata=DataFrame.from_query(context.dataset_info.sql),
            accumulate=[context.dataset_info.entity_key, target_name])
        
        predictions_df = pmml.result

    predictions_df.to_sql(table_name="predictions_tmp", if_exists="replace", temporary=True)

    metrics_df = DataFrame.from_query(f"""
    SELECT 
        HasDiabetes as y_test, 
        {byom_target_sql} as y_pred
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

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    # create confusion matrix plot
    cf = metrics.confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(cf, f"{context.artifact_output_path}/confusion_matrix")

    # calculate stats if training stats exist
    if os.path.exists(f"{context.artifact_input_path}/data_stats.json"):
        record_evaluation_stats(features_df=DataFrame.from_query(context.dataset_info.sql),
                                predicted_df=DataFrame.from_query("SELECT * FROM predictions_tmp"),
                                context=context)
