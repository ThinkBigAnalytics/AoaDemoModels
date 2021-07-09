import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn_pandas import DataFrameMapper
from teradataml import create_context, remove_context
from teradataml.dataframe.dataframe import DataFrame
from aoa.stats import stats
from aoa.util.artefacts import save_plot

import joblib
import os


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    feature_names = ["center_id", "meal_id", "checkout_price", 
                     "base_price", "emailer_for_promotion", "homepage_featured",
                     "category", "cuisine", "center_type", "op_area"]
    feature_names_cat = ["center_type", "category", "cuisine", "emailer_for_promotion", "homepage_featured"]
    target_name = "num_orders"
    
    print('Starting training ...')

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame(data_conf["table"])
    train_df = train_df.select([feature_names + [target_name]])
    train_pdf = train_df.to_pandas(all_rows = True)
    if "id" in train_df.columns:
        train_df.set_index("id", inplace=True)
    train_pdf[feature_names_cat] = train_pdf[feature_names_cat].astype("category")
    print('Loaded data ...')

    # split data into X and y
    X_train = train_pdf[feature_names]
    y_train = train_pdf[target_name]

    # modelling pipeline: feature encoding + algorithm
    oh_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    mapping = [(f, None) for f in feature_names if f not in feature_names_cat] + [
                                            (feature_names_cat, oh_encoder)]
    mapper = DataFrameMapper(mapping, input_df=True)
    regressor = RandomForestRegressor(random_state=hyperparams["rand_seed"],
                                      n_estimators=hyperparams["n_estimators"]
                                     )
    model = PMMLPipeline([("mapper", mapper),
                      ('regressor', regressor)])
    # preprocess training data and train the model
    model.fit(X_train, y_train)
    print("Finished training")

    # save feature names on pipeline for easy access later
    model.feature_names = feature_names
    model.feature_names_tr = mapper.transformed_names_ #feature names after transformation
    model.feature_names_cat = feature_names_cat
    model.target_name = [target_name]
    category_labels_overrides = {
        "emailer_for_promotion": {0: "false", 1: "true"},
        "homepage_featured": {0: "Not featured", 1: "Featured"}
    }
    model.category_labels_overrides = category_labels_overrides

    # export model artefacts
    print("Saving model to artifacts/output/model.joblib")
    joblib.dump(model, "artifacts/output/model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    print("Exporting model to artifacts/output/model.pmml")
    sklearn2pmml(model, "artifacts/output/model.pmml", with_repr = True)

    # save results and analytics
    print("Saving stats")
    feature_importances = model["regressor"].feature_importances_
    d = np.vstack((model.feature_names_tr, feature_importances)).T
    df_results = pd.DataFrame(d, columns=["Feature_names", "Weights"])
    df_results["Weights"] = df_results["Weights"].astype(float)
    df_results.sort_values(by="Weights", ascending=False, inplace=True)
    df_results[0:9].plot.bar(x="Feature_names", y="Weights")
    save_plot("feature_importance.png")

    # TODO: fix importance dictionary
    import shap
    mapper = model['mapper']
    shap_explainer = shap.TreeExplainer(model['regressor'])
    X_train = pd.DataFrame(mapper.transform(X_train), columns=model.feature_names_tr)
    X_shap = shap.sample(X_train, 100)
    shap_values = shap_explainer.shap_values(X_shap)
    feature_importances = pd.DataFrame(list(zip(model.feature_names_tr,
                                             np.abs(shap_values).mean(0))),
                                   columns=['col_name', 'feature_importance_vals'])
    feature_importances = feature_importances.set_index("col_name").T.to_dict(orient='records')[0]

    stats.record_training_stats(train_df,
                       features=model.feature_names,
                       predictors=model.target_name,
                       categorical=model.feature_names_cat,
                       importance=feature_importances,
                       category_labels=model.category_labels_overrides)

    remove_context()
    print("All done!")
