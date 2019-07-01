# Overview
Implement XGBoost algo for diabetes classification using Vantage ML engine via the teradataml python library.

# Datasets
The Dataset used is the PIMA diabestes dataset which can be downloaded from the internet. However, as the sample dataset is so small, we include it in this notesbooks/sample-data folder. The notebook will even import the dataset into Vantage for you so if you haven't setup the dataset before, start the notebook and import it.

In the AOA you will need to define the dataset metadata which is 

    {
        "hostname": "<vantage-db-url>",
        "data_tabe": "<the table with PIMA data>",
        "model_table": "<the table to write the model to>"
    }
    
Note the credentials are passed via environment variables which aligns with security on Kubernetes and other environments like this. This is of course configurable and depends on the organisations security requirements. If running in local mode, make sure the set the env variables. The two environment variables are 

    TD_USERNAME
    TD_PASSWORD


# Training
The [training.py](DOCKER/model_modules/training.py) 


# Evaluation
Evaluation is also performed in [scoring.evluate](DOCKER/model_modules/scoring.py) and
    

# Scoring 
The [scoring.py](DOCKER/model_modules/scoring.py) 

