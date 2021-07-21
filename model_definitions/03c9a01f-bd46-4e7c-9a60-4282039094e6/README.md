# Overview
Simple xgboost model python model based on the medical records for Pima Indians and whether or not each patient will have an onset of diabetes within ve years.

A sample notebook is located [here](notebooks/Explore%20Diabetes.ipynb).

# Datasets
The dataset required to train or evaluate this model is the PIMA Indians Diabetes dataset available [here](http://nrvis.com/data/mldata/pima-indians-diabetes.csv). The teradataml code to import it is

```python
import pandas as pd
from teradataml.dataframe.copy_to import copy_to_sql
from teradataml.dataframe.dataframe import DataFrame

df = pd.read_csv("http://nrvis.com/data/mldata/pima-indians-diabetes.csv", header=None)
df.columns = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age", "HasDiabetes"]

copy_to_sql(df = df, table_name = "PIMA", index=True, index_label="PatientId", if_exists="replace")

df = DataFrame("PIMA").sample(frac=[0.8, 0.2])

copy_to_sql(df = df[df.sampleid==1], table_name = "PIMA_TRAIN", index=False, if_exists="replace")
copy_to_sql(df = df[df.sampleid==2], table_name = "PIMA_TEST", index=False, if_exists="replace")
```


The dataset required to train or evaluate this model is the PIMA Indians Diabetes dataset available [here](http://nrvis.com/data/mldata/pima-indians-diabetes.csv).
This dataset is available in Teradata Vantage and already configured in the demo environment. For reference, the values which are required

Training
```json
{
    "table": "<training dataset>"
}
```
Evaluation

```json
{
    "table": "<test dataset>"
}
```

Batch Scoring
```json
 {
     "table": "<score dataset>",
     "predictions": "<ouput predictions dataset>"
 }
 ```


# Training
The [training.py](model_modules/training.py) produces the following artifacts

- model.joblib     (sklearn pipeline with scalers and xgboost model)
- model.pmml       (pmml version of the xgboost model and sklearn pipeline)

# Evaluation
Evaluation is defined in the `evaluate` method in [scoring.py](model_modules/scoring.py) and it returns the following metrics

- Accuracy
- Recall
- Precision
- f1-score

We produce a number of plots for each evaluation also

- roc curve
- confusion matrix

We also use shap to save some global explainability plots which help understand the importance and contribution of each feature

- shap feature importance


# Scoring 
This demo mode supports two types of scoring

 - Batch
 - RESTful
 - In-Vantage (IVSM)

In-Vantage scoring is supported via the PMML model we produce during scoring.

Batch Scoring is supported via the `score` method in [scoring.py](model_modules/scoring.py). As Evaluation is batch score + compare, the scoring logic is already validated with the evaluation step. The results of batch scoring are stored in the predictions table defined in the dataset template under `scoring` scope. 

The following table must exist to write (append) the scores into

```sql
CREATE MULTISET TABLE ivsm_pima_predictions, FALLBACK ,
     NO BEFORE JOURNAL,
     NO AFTER JOURNAL,
     CHECKSUM = DEFAULT,
     DEFAULT MERGEBLOCKRATIO,
     MAP = TD_MAP1
     (
        job_id VARCHAR(255),
        patient_id BIGINT, 
        score_result CLOB(2097088000) CHARACTER SET LATIN
     )
     PRIMARY INDEX ( job_id );
```

And the following view must exist to extract the specific prediction from the json output of IVSM.

```sql
CREATE VIEW ivsm_pima_predictions_v AS 
    SELECT job_id, patient_id, CAST(CAST(score_result AS JSON).JSONExtractValue('$.predicted_HasDiabetes') AS INT) as HasDiabetes 
    FROM ivsm_pima_predictions;
```

RESTful scoring is supported via the `ModelScorer` class which implements a predict method which is called by the RESTful Serving Engine. An example request is  

    curl -X POST http://<service-name>/predict \
        -H "Content-Type: application/json" \
        -d '{
            "data": {
                "ndarray": [
                        6,
                        148,
                        72,
                        35,
                        0,
                        33.6,
                        0.627,
                        50
                ]
            }
        }' 
