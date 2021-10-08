#TD/VAL libraries and VAL installation path
from teradataml import DataFrame, create_context, remove_context
from teradataml.analytics.Transformations import OneHotEncoder
from teradataml.analytics.Transformations import Retain
from teradataml import valib
from teradataml import configure
configure.val_install_location = "VAL"

import os

def train(data_conf, model_conf, **kwargs):
    """Python train method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """

    hyperparams = model_conf["hyperParameters"]
    
    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database="AOA_DEMO")

    ########################
    # load data & engineer #
    ########################
    table_name = data_conf["data_table"]
    numeric_columns = data_conf["numeric_columns"]
    target_column = data_conf["target_column"]
    categorical_columns = data_conf["categorical_columns"]
    
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
    df_train = tf.result

    # to avoid multi-collinearity issue we need to pass 
    # k-1 categories for each categorical feature to LinReg function
    excluded_cols = [target_column]
    for index, feature in enumerate(categorical_columns):
        ohe = one_hot_encode[index]
        f_name = ohe.values[-1] + "_" + feature
        excluded_cols.append(f_name)
    features = [col_name for col_name in df_train.columns if not col_name in excluded_cols]

    ##############################    
    # fit model to training data #
    ##############################
    print("Starting training...")
    model = valib.LinReg(data=df_train,
                         columns=features,
                         response_column=target_column,
                         entrance_criterion=hyperparams["entrance_criterion"],
                         use_fstat=hyperparams["use_fstat"],
                         use_pstat: hyperparams["use_pstat"]   
                        )

    print("Finished training")

    # saving model dataframes in the database so it could be used for evaluation and scoring    
    model.model.to_sql(table_name=kwargs.get("model_table"), if_exists='replace')
    model.statistical_measures.to_sql(table_name=kwargs.get("model_table") + "_rpt", if_exists='replace')

    print("Saved trained model")
    
    remove_context()

