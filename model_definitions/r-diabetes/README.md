# R Diabetes Prediction
## Overview
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
    
       
SELECT * 
FROM PIMA_PATIENT_FEATURES F JOIN PIMA_PATIENT_DIAGNOSES D
    ON F.patientid = D.patientid
    WHERE D.patientid MOD 5 <> 0
    
    
SELECT * 
FROM PIMA_PATIENT_FEATURES F JOIN PIMA_PATIENT_DIAGNOSES D
    ON F.patientid = D.patientid
    WHERE D.patientid MOD 5 = 0
```

    
## Training
The [training.R](model_modules/training.R) produces the following artifacts

- model.rds     (gbm parameters)

## Evaluation
Evaluation is also performed in [scoring.R](model_modules/scoring.R) by the function `evaluate` and it returns the following metrics

    accuracy: <acc>

## Scoring
The [scoring.R](model_modules/scoring.R) loads the model and metadata and accepts the dataframe for prediction.

### Batch mode
In this example, the values to score are in the table 'PIMA_TEST' at Teradata Vantage. The results are saved in the table 'PIMA_PREDICTIONS'. When batch deploying, this custom values should be specified:
   
   | key | value |
   |----------|-------------|
   | table | PIMA_TEST |
   | predictions | PIMA_PREDICTIONS |

### RESTful Sample Request

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

