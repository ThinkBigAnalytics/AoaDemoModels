# Overview
Simple xgboost model python model based on the medical records for Pima Indians and whether or not each patient will have an onset of diabetes within ve years.

# Datasets
The dataset required to train or evaluate this model is the PIMA Indians Diabetes dataset available [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv). The dataset descriptor is 

    {
        "data_path": "<path to csv>"
    }

# Training
The [training.R](DOCKER/model_modules/training.py) produces the following artifacts

- model.pkl     (xgboost pikle file with mode)

# Evaluation
Evaluation is also performed in [scoring.evluate](DOCKER/model_modules/scoring.py) and it returns the following metrics

    accuracy: <acc>

# Scoring 
The [scoring.R](DOCKER/model_modules/scoring.R) is responsible loads the model and metadata and accepts the dataframe for
for prediction. 

# Sample Request

    curl -X POST -H "Content-Type: application/json" -d "@data.json" http://<service-name>/predict
