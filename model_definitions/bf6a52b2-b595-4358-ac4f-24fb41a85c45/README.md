# R Diabetes Prediction
## Overview
PIMA Diabetes demo model using R

## Datasets
The dataset required to train or evaluate this model is the PIMA Indians Diabetes dataset available [here](http://nrvis.com/data/mldata/pima-indians-diabetes.csv). The dataset descriptor is 

    {
        "url": "http://nrvis.com/data/mldata/pima-indians-diabetes.csv",
        "test_split": 0.2
    }
    
## Training
The [training.R](model_modules/training.R) produces the following artifacts

- model.rds     (gbm parameters)

## Evaluation
Evaluation is also performed in [scoring.R](model_modules/scoring.R) by the function `evaluate` and it returns the following metrics

    accuracy: <acc>

## Scoring
The [scoring.R](model_modules/scoring.R) loads the model and metadata and accepts the dataframe for prediction.

### Batch mode
In this example the output is sent to stdout, but it could be saved into a table, file, etc.

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

