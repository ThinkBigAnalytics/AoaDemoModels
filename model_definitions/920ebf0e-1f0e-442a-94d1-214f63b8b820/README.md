# Overview
Implement XGBoost algo for diabetes classification using Vantage ML engine via the teradataml python library.

# Datasets
The Dataset used is the PIMA diabestes dataset which can be downloaded from the internet. However, as the sample dataset is so small, we include it in this notesbooks/sample-data folder. The notebook will even import the dataset into Vantage for you so if you haven't setup the dataset before, start the notebook and import it.

In the AOA you will need to define the dataset metadata which is 

    {
        "hostname": "<vantage-db-url>",
        "data_table": "<the table with PIMA data>",
        "model_table": "<the table to write the model to>"
    }
    
Note the credentials are passed via environment variables which aligns with security on Kubernetes and other environments like this. This is of course configurable and depends on the organisations security requirements. If running in local mode, make sure the set the env variables. The two environment variables are 

    TD_USERNAME
    TD_PASSWORD


# Training
The [training.py](DOCKER/model_modules/training.py) 


# Evaluation
Evaluation is also performed in [scoring.evluate](DOCKER/model_modules/scoring.py) in the evaluate method. Currently we only return accuracy but we could update it to do a confusion matrix also.

Due to a big with the teradataml library support for CLOBs, we currently only store the model in a models table in the database. The code to support exporting it and uploading to the model artefact repository of choice is also present, just disabled until this bug is resolved. The table schema required by the aoa is as follows and it inserts new records for every model version.

    CREATE SET TABLE aoa_models, NO FALLBACK
         (
          model_version VARCHAR(36) CHARACTER SET UNICODE,
          tree_id INTEGER,
          iter INTEGER,
          class_num INTEGER,
          tree CLOB(10485760) CHARACTER SET UNICODE,
          region_prediction CLOB(10485760) CHARACTER SET UNICODE)
    PRIMARY INDEX ( model_version );

    

# Scoring 
TBD what we want for scoring with Vantage models. Simply run the scoring on deploy or something else? 

