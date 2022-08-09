# Overview
PIMA Diabetes demo model using R

# Datasets
The dataset required to train or evaluate this model is the PIMA Indians Diabetes dataset available [here](http://nrvis.com/data/mldata/pima-indians-diabetes.csv). The teradataml code to import it is

```python
import pandas as pd
from teradataml import copy_to_sql

df = pd.read_csv("http://nrvis.com/data/mldata/pima-indians-diabetes.csv", header=None)
df.columns = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age", "HasDiabetes"]

copy_to_sql(df = df, table_name = "PIMA", index=True, index_label="PatientId", if_exists="replace")
```

```sql
CREATE TABLE PIMA_PATIENT_FEATURES AS 
    (SELECT 
        patientid,
        numtimesprg, 
        plglcconc, 
        bloodp, 
        skinthick, 
        twohourserins, 
        bmi, 
        dipedfunc, 
        age 
    FROM PIMA 
    ) WITH DATA;
    
    
CREATE TABLE PIMA_PATIENT_DIAGNOSES AS 
    (SELECT 
        patientid,
        hasdiabetes
    FROM PIMA 
    ) WITH DATA;
```

    
# Training
The [training.R](model_modules/training.R) produces the following artifacts

- model.rds     (gbm parameters)

# Evaluation
Evaluation is defined in the `evaluate` method in [scoring.R](model_modules/scoring.R) and it returns the following metrics

- Accuracy
- Recall
- Precision
- f1-score

We produce a number of plots for each evaluation also

- confusion matrix

# Scoring
This demo mode supports two types of scoring

 - Batch
 - RESTful

Batch Scoring is supported via the `score` method in [scoring.R](model_modules/scoring.R).  

The following table must exist to write (append) the scores into

```sql
CREATE MULTISET TABLE pima_patient_predictions (
    job_id VARCHAR(255), -- comes from airflow on job execution
    PatientId BIGINT,    -- entity key as it is in the source data
    HasDiabetes BIGINT,   -- if model automatically extracts target 
    json_report CLOB(1048544000) CHARACTER SET UNICODE  -- output of 
 )
 PRIMARY INDEX ( job_id );
```

RESTful scoring is supported via the `score.restful` function in [scoring.R](model_modules/scoring.R) which implements a predict method which is called by the RESTful Serving Engine. An example request is  

    curl -X POST http://localhost:5000/predict \
            -H "Content-Type: application/json" \
            -d '{
                "data": {
                    "ndarray": [[
                            6,
                            148,
                            72,
                            35,
                            0,
                            33.6,
                            0.627,
                            50
                    ]],
                    "names":[
                        "NumTimesPrg", 
                        "PlGlcConc", 
                        "BloodP", 
                        "SkinThick", 
                        "TwoHourSerIns", 
                        "BMI", 
                        "DiPedFunc", 
                        "Age"
                    ]
                }
            }' 

