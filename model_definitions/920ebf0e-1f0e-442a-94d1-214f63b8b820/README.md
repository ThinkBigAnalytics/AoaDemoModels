# Overview
Implement XGBoost algo for diabetes classification using Vantage ML engine via the teradataml python library.

# Datasets
The Dataset used is the PIMA diabestes dataset which can be downloaded from the internet. However, as the sample dataset is so small, we include it in this notesbooks/sample-data folder. The [sample notebook](notebooks/Explore%20Diabetes%20Vantage.ipynb) will even import the dataset into Vantage for you so if you haven't setup the dataset before, start the notebook and import it.

In the AOA you will need to define two datasets. One for training and evaluation.

Training (see [sample](.cli/datasets/train.json))

    {
        "hostname": "<vantage-db-url>",
        "data_table": "<the table with training PIMA data>"
    }
    

Evaluation (see [sample](.cli/datasets/evaluate.json))
  
    {
        "hostname":  "<vantage-db-url>",
        "data_table": "<the table with testing/hold-out PIMA data>",
        "predictions_table": "<the table to store the predictions>"
    }
    
Note the credentials are passed via environment variables which aligns with security on Kubernetes and other environments like this. This is of course configurable and depends on the organisations security requirements. If running in local mode, make sure the set the env variables. The two environment variables are 

    TD_USERNAME
    TD_PASSWORD


# Training
The [training.py](model_modules/training.py) is a simple XGBoost MLE model. 

Due to a bug with the teradataml library support for CLOBs, we currently only store the model in a models table in the database instead of exporting it and uploading to the model artefact repository. 

We store the models in a unique table for each model version. It is based on the first part of the model version id and we prepend the string `AOA_MODELS_` to make it clear the table is a model version managed by the AOA. An example table name is `AOA_MODELS_ba80ac47` where the model version is `ba80ac47-f868-449d-93e8-abf601fce8aa`. Evaluations and subsequent scoring in the MLE can use this table name then as it is associated with the trained model version.


# Evaluation
Evaluation is also performed in the [evaluation code](model_modules/scoring.py) in the evaluate method. We record the statistics from the confusion matrix support in the MLE. You could store more statistics but this is all we store at the moment. Note that you can store any key-value pairs you want. We would like to add support to the UI to display the actual confusion matrix correctly. 

# Scoring 
As mentioned in the training section, we are evaluating the possibility to export the MLE models in a teradata supported format (AML) and score them in RESTful APIs. This will allow for low latency use cases such as Next Best Offer. We are also evaluating if the 4 supported model types in the SQLE are capable of this individual offer prediction at a low latency high QPS.

The majority of models will be scored as a Batch prediction. This means they will either be executed in the SQLE if it is one of the 4 supported models, or in the MLE for the other model types. We provide the scoring method which is used by the evaluation and whatever scheduler is chosen by the organisation would simply invoke this. As the scheduler is organisation specific we don't provide anything for it, but integration is simple. We may look to provide a default Airflow demo or similiar. 

