"""
AOPS"s train function implementation for 
the demand forecasting regression model
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from nyoka import skl_to_pmml
from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from aoa.stats import stats
from aoa.util.artefacts import save_plot

import joblib
import os


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"])

    feature_names = ["center_id", "meal_id", "checkout_price", 
                     "base_price", "emailer_for_promotion", "homepage_featured",
                     "category", "cuisine", "center_type", "op_area"]
    feature_names_cat = ["center_type", "category", "cuisine"]
    target_name = "num_orders"
    
    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame(data_conf["table"])
    train_df = train_df.select([feature_names + [target_name]])
    train_pdf = train_df.to_pandas(all_rows = True)
    if "id" in train_df.columns:
        train_df.set_index("id", inplace=True)
    train_pdf[feature_names_cat] = train_pdf[feature_names_cat].astype("category")

    # split data into X and y
    X_train = train_pdf[feature_names]
    y_train = train_pdf[target_name]

    # encode categorical features 
    oh_encoder = make_column_transformer(
        (OneHotEncoder(sparse=False, handle_unknown="ignore"),
         make_column_selector(dtype_include="category")),
        remainder="passthrough")
    model = make_pipeline(oh_encoder,
                    RandomForestRegressor(random_state=hyperparams["rand_seed"],
                                          n_estimators=hyperparams["n_estimators"]
                                          ))
    # preprocess training data and train the model
    model.fit(X_train, y_train)
    print("Finished training")

    # save feature names on pipeline for easy access later
    ct = model['columntransformer']
    model.feature_names = feature_names
    model.feature_names_tr = ct.get_feature_names() #feature names after transformation
    model.feature_names_cat = feature_names_cat
    model.target_name = target_name
    cat_feature_dict = {}
    for feature in feature_names_cat:
        cat_feature_dict[feature] = dict(enumerate(train_pdf[feature].
                                                   cat.categories ))
    model.cat_feature_dict = cat_feature_dict

    # export model artefacts
    joblib.dump(model, "artifacts/output/model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    skl_to_pmml(pipeline=model, col_names=feature_names, 
                target_name=target_name, 
                pmml_f_name="artifacts/output/model.pmml")
    print("Saved trained model")

    # save results and analytics
    feature_names_tr = model.feature_names_tr
    feature_importances = model["randomforestregressor"].feature_importances_
    d = np.vstack((feature_names_tr, feature_importances)).T
    df_results = pd.DataFrame(d, columns=["Feature_names", "Weights"])
    df_results["Weights"] = df_results["Weights"].astype(float)
    df_results.sort_values(by="Weights", ascending=False, inplace=True)
    df_results[0:9].plot.bar(x="Feature_names", y="Weights")
    save_plot("feature_importance.png")
    stats.record_stats(train_df,
                       features=feature_names_tr,
                       predictors=target_name,
                       categorical=feature_names_cat,
                       importance=feature_importances,
                       category_labels=cat_feature_dict)