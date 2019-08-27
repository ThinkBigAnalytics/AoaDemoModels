- [Models](#models)
- [Available Models](#available-models)
- [Adding new Models](#adding-new-models)
  * [Python Model Signatures](#python-model-signatures)
    + [Python Training Progress Callback](#python-training-progress-callback)
    + [Shared Code](#shared-code)
  * [R Model Signatures](#r-model-signatures)
- [Model Container and Resources Configuration](#model-container-and-resources-configuration)
  * [Resource Requests](#resource-requests)
  * [Configure Base Docker Image](#configure-base-docker-image)
- [Cli tools](#cli-tools)
  * [Running Models Locally](#running-models-locally)
  * [Adding Models based on Templates](#adding-models-based-on-templates)

# Models

This repository contains demo models for the Teradata AOA.

The data science development perspective and process can be summarised by the following diagram

<center><img src="./images/ds-perspective.png" style="width: 1024px"/></center>

# Available Models

- [Tensorflow Sentitment Analysis](model_definitions/74eca506-e967-48f1-92ad-fb217b07e181)
- [Python XGboost (Diabetes)](model_definitions/03c9a01f-bd46-4e7c-9a60-4282039094e6)
- [Python MLE XGboost (Diabetes)](model_definitions/920ebf0e-1f0e-442a-94d1-214f63b8b820)
- [SQL Naive Bayes (Churn)](model_definitions/78036940-248e-4dae-b690-6f70eefcb188)
- [SQL SVM (Churn)](model_definitions/78036940-248e-4dae-b690-6f70eefcb188)
- [Demo R GBM](model_definitions/bf6a52b2-b595-4358-ac4f-24fb41a85c45)
- [PySpark Logistic Regression](model_definitions/149e31ed-c554-46b4-95d2-00c5c43320fb)

# Adding new Models

To add a new model, simply use the repo cli tool which helps to create the structure necessary for the given model. See [here](#adding-models-based-on-templates). 

    ./cli/repo-cli.py -a
    
Note that you should manually add the new model to the [Available Models](#available-models) table above so that you can quickly access it from the main repository page. The cli tool will eventually update this as part of the process of adding a new repo. 

The models are all defined under the directory [model_definitions](./model_definitions). 

## Python Model Signatures

The python train and evaluate functions are 

    def train(data_conf, model_conf, **kwargs):
       ...
       
    def evaluate(data_conf, model_conf, **kwargs):
       ...
       
If deploying behind a Restful scoring engine, the predict function is declared within a ModelScorer class
       
    class ModelScorer(object):
        ...
        
        def predict(self, data):
            ...

Note that the `**kwargs` is used to ensure future extendibility of the model management framework and ensuring that models are backward compatible. For instance, we may pass new features such as callback handlers that the frameworks supports to newer models, but the old models can safely ignore such parameters.

The folder structure for a python model is 

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
    
And if you want to deploy this behind a restful API add the ModelScorer abstraction for the REST engine

    from .scoring import ModelScorer
    

### Python Training Progress Callback

We have added an example of the training progress support in the AOA framework to the [Tensorflow Sentitment Analysis](model_definitions/74eca506-e967-48f1-92ad-fb217b07e181/DOCKER/model_modules/callback.py). This works by sending progress messages via activemq which can in take whatever action it needs to based on the progress, update a UI for example or trigger some alert if a progress is stalled. The provided `AoaKerasProgressCallback` function should be located in an AOA python module whenever this is created and simply imported instead of being defined in the sample model.

### Shared Code

To share common code between models, you should create a separate models util module that you can release and version. To share code between the training and scoring of an individual model, you can follow the example code in the [tensorflow sample](model_definitions/74eca506-e967-48f1-92ad-fb217b07e181/DOCKER/model_modules). There you can see that the [preprocess.py](model_definitions/74eca506-e967-48f1-92ad-fb217b07e181/DOCKER/model_modules/preprocess.py) file has code that is used in both training.py and scoring.py. 


## R Model Signatures

The R model signatures are exactly the same as the python signatures, with the `... being the equivalent of python `**kwargs`

    train <- function(data_conf, model_conf, ...) {
       ...
    }
    
    evaluate <- function(data_conf, model_conf, ...) {
       ...
    }
    
If deploying behind a Resful engine, the predict method should also be declared as follows

    predict.model <- function(model, data) {
    
    }
    
In the case of an R model, the model_modules folder is slightly different with

    model_modules/
        requirements.R
        scoring.R
        training.R

Note that other files may be included under the model_modules folder and they will be added to the relevant containers during training, evaluation and scoring. Examples of this a common data prep classes etc.


# Model Container and Resources Configuration

## Resource Requests
You can configure the kubernetes resources requested for each model on a per model basis. This allows to limit the resource usage but also to request things like gpus etc. If nothing is specified, then no limits are currently applied, however we will eventually apply defaults. To do this, just add the following to the model.json for the given model.

    "resources": {
      "training": {
        "limits": {
          "cpu": "200m",
          "memory": "100Mi"
        }
      }
    }
    
If your model is a spark model, things change slightly. You simply specify the args you want to pass to spark as follows. Note that the master option allows testing locally but should be set to yarn in a deployment setup. 

    "resources": {
        "training": {
            "master": "local[1]",
            "args": "--num-executors 1 --executor-cores 1 --driver-memory 1G --executor-memory 1G"
        }
    }

## Configure Base Docker Image
We also support specifying per model base docker images to use in training and evaluate. Eventually this will also apply for scoring but for now, we only support specifying a base image per model for training and evaluation. This is very useful to have dependencies preinstalled for faster model training and also to have additional gpu drivers or dependencies installed in the case of using gpus. To do this, just add the following to the model.json for the given model. 

    "docker": {
        "trainerImage": "willfleury/r_trainer:2.9"
    }


# Cli tools

## Running Models Locally
To aid developing and testing the models setup in the AOA locally and in the datalab, we provide some useful cli tools to 
run the model training and evaluation using the config and data that you expect to be passed during automation.

Python

    ./cli/run-model-cli.py
 

R

    TBC..
 
 
## Adding Models based on Templates

You can add models based using a cli tool based on model templates defined in the [model_templates](./model_templates) folder. 

    ./cli/repo-cli.py -a
