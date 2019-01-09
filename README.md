# CaixaModels

This repository contains the sample models for the Caixa AI review project.

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
                
In the case of an R model, the model_modules folder is slightly different with

                model_modules/
                    requirements.R
                    scoring.R
                    training.R

Note that other files may be included under the model_modules folder and they will be added to the relevant containers during training, evaluation and scoring. Examples of this a common data prep classes etc.

# Shared Code

To share common code between models, or between training and scoring, you should create a separate models util module that you can release and version. It should be relatively easy to support relative files within the model_modules folder so that at the very least, training and scoring in the same packages can have common code in a shared file without the need for a library with additional release cycle and maintenance. It just needs some testing. 

# Cli tools

To aid developing and testing the models setup in the AOA locally and in the datalab, we provide some useful cli tools to 
run the model training and evaluation using the config and data that you expect to be passed during automation.

Python
 - [local-model-cli.py](./cli/local-model-cli.py)

R
 - TBC..
