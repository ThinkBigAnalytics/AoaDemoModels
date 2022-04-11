from teradataml import copy_to_sql, DataFrame
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)

import joblib
import pandas as pd


def score(context: ModelContext):

    aoa_create_context()

    model = joblib.load(f"{context.artefact_input_path}/model.joblib")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    features_tdf = DataFrame.from_query(context.dataset_info.sql)
    features_pdf = features_tdf.to_pandas(all_rows=True)

    print("Scoring")
    predictions_pdf = model.predict(features_pdf)

    print("Finished Scoring")

    # store the predictions
    predictions_pdf = pd.DataFrame(predictions_pdf, columns=[target_name])
    predictions_pdf["PatientId"] = features_pdf["PatientId"].values
    predictions_pdf["job_id"] = context.job_id

    copy_to_sql(df=predictions_pdf,
                schema_name=context.dataset_info.prediction_database,
                table_name=context.dataset_info.prediction_table,
                index=False,
                if_exists="append")

    # calculate stats
    predictions_df = DataFrame.from_query(f"""
        SELECT 
            * 
        FROM {context.dataset_info.prediction_database}.{context.dataset_info.prediction_table} 
            WHERE job_id = '{context.job_id}'
    """)

    record_scoring_stats(features_tdf, predictions_df)


# Add code required for RESTful API
class ModelScorer(object):

    def __init__(self):
        self.model = joblib.load("artifacts/input/model.joblib")

    def predict(self, data):
        return self.model.predict(data)
