# Overview
Simple xgboost model python model based on the medical records for Pima Indians and whether or not each patient will have an onset of diabetes within ve years.

A sample notebook is located [here](notebooks/Explore%20Diabetes.ipynb).

# Datasets
The dataset required to train or evaluate this model is the PIMA Indians Diabetes dataset available [here](http://nrvis.com/data/mldata/pima-indians-diabetes.csv). The dataset descriptor is 

    {
        "url": "http://nrvis.com/data/mldata/pima-indians-diabetes.csv",
        "test_split": 0.2
    }
    
For Batch scoring, we also include the [dataset_template.json](./scheduler/dataset_template.json) which is required by the Airflow (or other) scheduler. This will not be necessary anymore after 2.7+ of the AOA as the user will select the dataset template when deploying.

# Training
The [training.py](model_modules/training.py) produces the following artifacts

- model.pkl     (xgboost pickle file with mode)
- scaler.pkl    (the scaler file)

# Evaluation
Evaluation is defined in the `evaluate` method in [scoring.py](model_modules/scoring.py) and it returns the following metrics

    Accuracy
    Recall
    Precision
    f1-score

# Scoring 
This demo mode supports two types of scoring

 - Batch
 - RESTful
 
Batch Scoring is supported via the `score` method  [scoring.py](model_modules/scoring.py). As Evaluation is batch score + compare, the scoring logic is alreayd validated with the evaluation step. As batch scoring is run by a scheduler, the scheduler must know how to tell it what dataset (`data_conf`) it should execute on. It does this by using the `scheduler/dataset_template.json` which is templated by the scheudler (airflow in the demo) with things such as dates which are necessary. This will not be necessary anymore after 2.7+ of the AOA as the user will select the dataset template when deploying.

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
