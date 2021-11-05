#TD/VAL libraries and VAL installation path
from teradataml import DataFrame, create_context, remove_context
from teradataml.analytics.Transformations import OneHotEncoder
from teradataml.analytics.Transformations import Retain
from teradataml import valib
from teradataml import configure
configure.val_install_location = "VAL"
from aoa.util.artefacts import save_plot
from aoa.stats import stats


import os
import json
#import sklearn.metrics as skm
import pandas as pd
import numpy as np


def evaluate(data_conf, model_conf, **kwargs):
    """Python evaluate method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """
    create_context(host = os.environ["AOA_CONN_HOST"],
                   username = os.environ["AOA_CONN_USERNAME"],
                   password = os.environ["AOA_CONN_PASSWORD"],
                   database = "AOA_DEMO")

    ########################
    # load data & engineer #
    ########################
    table_name = data_conf["data_table"]
    numeric_columns = ["checkout_price","base_price","emailer_for_promotion","homepage_featured","op_area"]
    categorical_columns = ["center_type","category","cuisine"]
    target_column = "num_orders"
    features = numeric_columns + categorical_columns


    """
    The eval/scoring data is encoded here similarly as done 
    with the training data. The assumption is that all categorical variables
    contain same categories in both the training and eval/score datasets.
    However, if this assumption does not hold or there is uncertainty then it is 
    best to save and carry forward all the distinct categorical feature values to 
    eval/score model codes and use those for encoding. 

    The default VAL's OHE behavior is to ignore any categories not listed in the encoder
    definition call, by having an all zero record (for all the categorical features). If 
    this behavior is undesired then a simple technique to filter out the records with new
    categories can be adopted as follows:

    df = DataFrame(table_name)
    for feature in categorical_columns:
       df = df[~df[feature].isin(cat_feature_values[feature])]

    """

    # feature encoding
    # categorical features to one_hot_encode using VAL transform
    cat_feature_values = {}
    for feature in categorical_columns:
        #distinct has a spurious behaviour so using Group by
        q = 'SELECT ' + feature + ' FROM ' + table_name + ' GROUP BY 1;'  
        df = DataFrame.from_query(q)
        cat_feature_values[feature] = list(df.dropna().get_values().flatten())

    one_hot_encode = []
    for feature in categorical_columns:
        ohe = OneHotEncoder(values=cat_feature_values[feature], columns=feature)
        one_hot_encode.append(ohe)

    # carried forward columns using VAL's Retain function
    retained_cols = numeric_columns+[target_column]
    retain = Retain(columns=retained_cols)    

    data = DataFrame(data_conf["data_table"])
    tf = valib.Transform(data=data, one_hot_encode=one_hot_encode, retain=retain)
    df_eval = tf.result

    # evaluate using fitted model on evaluation data #
    df_score_eval = valib.LinRegEvaluator(data=df_eval,
                                  model=DataFrame(kwargs.get("model_table"))
                                  )
    eval_result = df_score_eval.result
    evaluation = {}
    for col in eval_result.columns:
        evaluation[col] = eval_result.get(col).squeeze()

    with open("artifacts/output/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    model = valib.LinRegPredict(data=df_eval,
                        model=DataFrame(kwargs.get("model_table")),
                        accumulate=target_column
                        )
    df_score = model.result

    pdf_score = df_score.to_pandas(all_rows=True)
    y_test = pdf_score[target_column]
    y_pred = pdf_score["_tm_" + target_column]

    # a plot of actual num_orders vs predicted 
    result_df = pd.DataFrame(np.vstack((y_test, y_pred)).T, 
                                        columns=['Actual', 'Predicted'])
    df = result_df.sample(n=100, replace=True)
    df['No.'] = range(len(df))
    df.plot(x="No.", y=['Actual', 'Predicted'], kind = 'line', legend=True, 
            subplots = False, sharex = True, figsize = (5.5,4), ls="none", 
            marker="o", alpha=0.4)    
    save_plot('Actual vs Predicted')
    
    stats.record_evaluation_stats(data.select(features+[target_columnn]), df_score)

    print("Evaluation complete...")

    remove_context()
