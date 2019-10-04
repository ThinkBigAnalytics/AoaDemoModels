# Overview
Implement XGBoost algo for diabetes classification using Vantage ML engine via the teradataml python library.

# Datasets
The Dataset used is the PIMA diabestes dataset which can be downloaded from the internet. However, as the sample dataset is so small, we include it in this notesbooks/sample-data folder. The [sample notebook](notebooks/Explore%20Diabetes%20Vantage.ipynb) will even import the dataset into Vantage for you so if you haven't setup the dataset before, start the notebook and import it.

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
The [training.py](model_modules/training.py) is a simple XGBoost MLE model. 

Due to a big with the teradataml library support for CLOBs, we currently only store the model in a models table in the database instead of exporting it and uploading to the model artefact repository. The code to support exporting it and uploading to the model artefact repository of choice is also present, just disabled until this bug is resolved. 

We could easily add code to copy the model to a general models table which is keyed by the model version, however we prefer the approach of exporting as this allows us to convert to the AML format and score MLE models behind low latency RESTful APIs. This is something we are evaluating with the product team. 


# Evaluation
Evaluation is also performed in [scoring.evluate](model_modules/scoring.py) in the evaluate method. Currently we only return accuracy but we could update it to do a confusion matrix also.

# Scoring 
As mentioned in the training section, we are evaluating the possibility to export the MLE models in a teradata supported format (AML) and score them in RESTful APIs. This will allow for low latency use cases such as Next Best Offer. We are also evaluating if the 4 supported model types in the SQLE are capable of this individual offer prediction at a low latency high QPS.

The majority of models will be scored as a Batch prediction. This means they will either be executed in the SQLE if it is one of the 4 supported models, or in the MLE for the other model types. We provide the scoring method which is used by the evaluation and whatever scheduler is chosen by the organisation would simply invoke this. As the scheduler is organisation specific we don't provide anything for it, but integration is simple. We may look to provide a default Airflow demo or similiar. 

