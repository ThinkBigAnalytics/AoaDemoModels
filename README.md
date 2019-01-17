# CaixaModels

This repository contains the sample models for the Caixa AI review project.

The data science development perspective and process can be summarised by the following diagram

<center><img src="./images/ds-perspective.png" style="width: 1024px"/></center>

# Model Definitions

The models are all defined under the directory [model_definitions](./model_definitions). The directory structure is as follows:

    model_definitions/
        <model_id>/
            README.md
            model.json
            <aoa-trainer-framework>/
                config.json
                model_modules/
                    __init__.py
                    requirements.txt
                    scoring.py
                    training.py

The __init__.py must be defined in a specific way to import the training and scoring modules. This is important to support relative imports and shared code for a model.

    from .training import train
    from .scoring import evaluate
                
In the case of an R model, the model_modules folder is slightly different with

                model_modules/
                    requirements.R
                    scoring.R
                    training.R

Note that other files may be included under the model_modules folder and they will be added to the relevant containers during training, evaluation and scoring. Examples of this a common data prep classes etc.

# Shared Code

To share common code between models, you should create a separate models util module that you can release and version. To share code between the training and scoring of an individual model, you can follow the example code in the [tensorflow sample](./model_definitions/74eca506-e967-48f1-92ad-fb217b07e181/DOCKER_GENERIC_RAW/model_modules). There you can see that the [preprocess.py](./model_definitions/74eca506-e967-48f1-92ad-fb217b07e181/DOCKER_GENERIC_RAW/model_modules/preprocess.py) file has code that is used in both training.py and scoring.py. 

# Cli tools

To aid developing and testing the models setup in the AOA locally and in the datalab, we provide some useful cli tools to 
run the model training and evaluation using the config and data that you expect to be passed during automation.

Python
 - [local-model-cli.py](./cli/local-model-cli.py)

R
 - TBC..
