#import pandas as pd
#VAL and teradata ML Libraries
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
    
    create_context(host = os.environ["AOA_CONN_HOST"],
                   username = os.environ["AOA_CONN_USERNAME"],
                   password = os.environ["AOA_CONN_PASSWORD"],
                   database = "EP_SDS")

    ########################
    # load data & engineer #
    ########################
    data = DataFrame(data_conf["data_table"])
    #we can use ML"s OneHotEncoder to transform the x variable so it can be treated as numeric
    centers = ["TYPE_A", "TYPE_B", "TYPE_C"]
    cuisines = ["Continental", "Indian", "Italian", "Thai"]
    meals = ["Beverages", "Biryani", "Desert", "Extras", "Fish", "Other Snacks", "Pasta", 
             "Pizza", "Rice Bowl", "Salad", "Sandwich", "Seafood", "Soup", "Starters"]
    ohe_center = OneHotEncoder(values=centers, columns= "center_type")
    ohe_cuisine = OneHotEncoder(values=cuisines, columns= "cuisine")
    ohe_meal = OneHotEncoder(values=meals, columns= "category")
    one_hot_encode = [ohe_center, ohe_cuisine, ohe_meal]
    
    retained_cols = ["center_id", "meal_id", "checkout_price", "base_price",
           "emailer_for_promotion", "homepage_featured", "op_area", "num_orders"]
    retain = Retain(columns=retained_cols)
    
    tf = valib.Transform(data=data, one_hot_encode=one_hot_encode, retain=retain)
    df_train = tf.result

    print("Starting training...")

    ##############################    
    # fit model to training data #
    ##############################
    # to avoid multi-collinearity issue we need to pass 
    # k-1 categories for each categorical feature to LinReg function
    features = [col_name for col_name in df_train.columns if not (col_name=="num_orders" 
                or col_name=="TYPE_C_center_type"
                or col_name=="Thai_cuisine"
                or col_name=="Starters_category")]
    model = valib.LinReg(data=df_train, 
                     columns=features, 
                     response_column="num_orders")

    print("Finished training")

    # saving model dataframes in the database so it could be used for evaluation and scoring
    
    model.model.to_sql(table_name = kwargs.get("model_table"), if_exists = 'replace')
    model.statistical_measures.to_sql(table_name = kwargs.get("model_table") + "_rpt", if_exists = 'replace')


    print("Saved trained model")
    
    remove_context()

