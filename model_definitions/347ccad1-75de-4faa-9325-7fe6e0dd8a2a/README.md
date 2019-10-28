# Overview
MLE DecisionForest function using teradataml taken from [here](https://docs.teradata.com/reader/GsM0pYRZl5Plqjdf9ixmdA/ZM0DHgB33mPUsom4ucZP0g)

# Datasets
The dataset are selected by the user during AOA implementation.  All datasets must exist within the Vantage environment specified in the data_conf prior to training/scoring in the AOA.

# Training
Training occurs via the Python teradataml package in the [training.py](model_modules/training.py) within Vantage and the model is stored under the given user with the model_table name specified in the data_conf.

# Evaluation 
Evaluation (Scoring) occurs via the Python teradataml package in the [scoring.py](model_modules/scoring.py) within Vantage.  The model_table from the Training data_conf is retrieved and the model is scored with a newly specified scoring data set in a subsequent data_conf.