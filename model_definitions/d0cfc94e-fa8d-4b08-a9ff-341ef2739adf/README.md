# Overview
Implement XGBoost algo for diabetes classification using Vantage ML engine via SQL. This is a replica of the python version of this model using the teradataml library which you can see [here](../920ebf0e-1f0e-442a-94d1-214f63b8b820).

# Datasets
The Dataset used is the PIMA diabestes dataset which can be downloaded from the internet. We include details of how to import this dataset into Vantage in the notebook under the python demo [here](../920ebf0e-1f0e-442a-94d1-214f63b8b820/notebooks/Explore%20Diabetes%20Vantage.ipynb).

In the AOA you will need to define two datasets. One for training and evaluation.

Training

    {
        "hostname": "<vantage-db-url>",
        "data_table": "<the table with training PIMA data>"
    }
    

Evaluation 
  
    {
        "hostname":  "<vantage-db-url>",
        "data_table": "<the table with testing/hold-out PIMA data>",
        "metrics_table": "<the table the evaluation metrics will be in>",
        "predictions_table": "<the table to store the predictions>"
    }
    
Note the credentials are passed via environment variables which aligns with security on Kubernetes and other environments like this. This is of course configurable and depends on the organisations security requirements. If running in local mode, make sure the set the env variables. The two environment variables are 

    TD_USERNAME
    TD_PASSWORD


# Training
The [training.sql](model_modules/training.sql) is a simple XGBoost MLE model. 

Due to a bug with the teradataml library support for CLOBs, we currently only store the model in a models table in the database instead of exporting it and uploading to the model artefact repository.

We store the models in a unique table for each model version. It is based on the first part of the model version id and we prepend the string `AOA_MODELS_` to make it clear the table is a model version managed by the AOA. An example table name is `AOA_MODELS_ba80ac47` where the model version is `ba80ac47-f868-449d-93e8-abf601fce8aa`. Evaluations and subsequent scoring in the MLE can use this table name then as it is associated with the trained model version. For the sql code, we provide the following variable which manages this name. 

    {{ model_table }}


# Evaluation
Evaluation is also performed in the [evaluation code](model_modules/scoring.sql) in the evaluate method. We record the statistics from the confusion matrix support in the MLE. You could store more statistics but this is all we store at the moment. Note that you can store any key-value pairs you want. We would like to add support to the UI to display the actual confusion matrix correctly. 

The "metrics_table" parameter in the dataset metadata is what tells the framework, and the sql code where to place the metrics it wants to surface and record for the given sql model. Therefore, your sql code should add all the metrics it wants surfaced into that table.

# Scoring 
For sql model types, we need to figure out how to separate this. Most likley we create a separate evaluation and scoring file for sql.. The other option is to use the templating support to exclude the evaluation block when we only want to score. 