# Overview
Simple xgboost model python model based on the medical records for Pima Indians and whether or not each patient will have an onset of diabetes within ve years.

A sample notebook is located [here](notebooks/Explore%20Diabetes.ipynb).

# Datasets
The dataset required to train or evaluate this model is the PIMA Indians Diabetes dataset available [here](http://nrvis.com/data/mldata/pima-indians-diabetes.csv). The dataset descriptor is 

    {
        "url": "http://nrvis.com/data/mldata/pima-indians-diabetes.csv",
        "test_split": 0.2
    }

# Training
The [training.py](model_modules/training.py) produces the following artifacts

- model.pkl     (xgboost pickle file with mode)
- scaler.pkl    (the scaler file)

# Evaluation
Evaluation is also performed in [scoring.evaluate](model_modules/scoring.py) and it returns the following metrics

    accuracy: <acc>

# Scoring 
The [scoring.py](model_modules/scoring.R) is responsible loads the model and metadata and accepts the dataframe for
for prediction. 

# Sample Request

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
