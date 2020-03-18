- [Models](#models)
- [Available Models](#available-models)
- [Adding new Models](#adding-new-models)
  - [Python Model Signatures](#python-model-signatures)
    - [Python Training Progress Callback](#python-training-progress-callback)
    - [Shared Code](#shared-code)
  - [R Model Signatures](#r-model-signatures)
  - [SQL Model Signatures](#sql-model-signatures)
  - [Additional Variables](#additional-variables)
- [Model Container and Resources Configuration](#model-container-and-resources-configuration)
  - [Resource Requests](#resource-requests)
  - [Configure Base Docker Image](#configure-base-docker-image)
- [CLI tools](#cli-tools)
  - [Install AOA Python Package](#install-aoa-python-package)
    - [Install from Python Package Index](#install-from-python-package-index)
    - [Install from Source Code](#install-from-source-code)
  - [Running Models Locally](#running-models-locally)
    - [Train](#train)
    - [Evaluate](#evaluate)
  - [Adding Models based on Templates](#adding-models-based-on-templates)

# Models

This repository contains demo models for the Teradata AOA. See the [HowTo](./HOWTO.md) guide for a brief tutorial on how to add new models.

# Available Models

More human friendly folder names for each model under model_definitions is tracked in [issue-105](https://github.com/ThinkBigAnalytics/AoaCoreService/issues/105).

- [Tensorflow Sentitment Analysis](model_definitions/74eca506-e967-48f1-92ad-fb217b07e181)
- [Python XGboost (Diabetes)](model_definitions/03c9a01f-bd46-4e7c-9a60-4282039094e6)
- [Python MLE XGboost (Diabetes)](model_definitions/920ebf0e-1f0e-442a-94d1-214f63b8b820)
- [SQL MLE XGboost (Diabetes)](model_definitions/d0cfc94e-fa8d-4b08-a9ff-341ef2739adf)
- [Demo R GBM](model_definitions/bf6a52b2-b595-4358-ac4f-24fb41a85c45)
- [PySpark Logistic Regression](model_definitions/149e31ed-c554-46b4-95d2-00c5c43320fb)

# Adding new Models

To add a new model, simply use the aoa python package cli tool which helps to create the structure necessary for the given model, see [here](#adding-models-based-on-templates).  You can install aoa python cli following the instructions specified [here](#install-aoa-python-package)

    aoa --add
    
Note that you should manually add the new model to the [Available Models](#available-models) table above so that you can quickly access it from the main repository page. The cli tool will eventually update this as part of the process of adding a new repo. 

The models are all defined under the directory [model_definitions](./model_definitions). 

Supported model types are 

| Language   |      Description      |
|----------|:-------------:|
| python |  [Python Model Signatures](#python-model-signatures) |
| R | [R Model Signatures](#r-model-signatures) |
| sql | [SQL Model Signatures](#sql-model-signatures) |


## Python Model Signatures

The python train and evaluate functions are 

    def train(data_conf, model_conf, **kwargs):
       ...
       
    def evaluate(data_conf, model_conf, **kwargs):
       ...

To know what values are passed in the varargs, see [Additional Variables](#additional-variables).

       
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
            config.json
            model_modules/
                __init__.py
                requirements.txt
                scoring.py
                training.py
            notebooks/

The __init__.py must be defined in a specific way to import the training and scoring modules. This is important to support relative imports and shared code for a model.

    from .training import train
    from .scoring import evaluate
    
And if you want to deploy this behind a restful API add the ModelScorer abstraction for the REST engine

    from .scoring import ModelScorer
    

### Python Training Progress Callback

We have added an example of the training progress support in the AOA framework to the [Tensorflow Sentitment Analysis](model_definitions/74eca506-e967-48f1-92ad-fb217b07e181/model_modules/callback.py). This works by sending progress messages via activemq which can in take whatever action it needs to based on the progress, update a UI for example or trigger some alert if a progress is stalled. The provided `AoaKerasProgressCallback` function should be located in an AOA python module whenever this is created and simply imported instead of being defined in the sample model.

### Shared Code

To share common code between models, you should create a separate models util module that you can release and version. To share code between the training and scoring of an individual model, you can follow the example code in the [tensorflow sample](model_definitions/74eca506-e967-48f1-92ad-fb217b07e181/model_modules). There you can see that the [preprocess.py](model_definitions/74eca506-e967-48f1-92ad-fb217b07e181/model_modules/preprocess.py) file has code that is used in both training.py and scoring.py. 


## R Model Signatures

The R model signatures are exactly the same as the python signatures, with the `... being the equivalent of python `**kwargs`

    train <- function(data_conf, model_conf, ...) {
       ...
    }
    
    evaluate <- function(data_conf, model_conf, ...) {
       ...
    }
    
To know what values are passed in the varargs, see [Additional Variables](#additional-variables).

If deploying behind a Resful engine, the predict method should also be declared as follows

    predict.model <- function(model, data) {
    
    }
    
In the case of an R model, the model_modules folder is slightly different with

    model_modules/
        requirements.R
        scoring.R
        training.R

Note that other files may be included under the model_modules folder and they will be added to the relevant containers during training, evaluation and scoring. Examples of this a common data prep classes etc.


## SQL Model Signatures

SQL Models obviously don't have a function signature. Instead, we approach the sql models slightly differently. You define training.sql and scoring.sql files where you place your sql logic for calling the Vantage training and evaluation functions. To support using different dataset metadata, different model configuration etc, we use templating. So, the data_conf argument that is available to the python and R models can be used in sql as 

    CREATE TABLE {{ data_conf.data_table }}
    AS (
    ...
    
and the model_conf can be used in a similar way,

    SELECT * FROM XGBoostPredict(
        	...
        	ON "{{ data_conf.model_table }}" AS ModelTable
        	DIMENSION
        	ORDER BY "tree_id","iter","class_num"
        	USING
        	IdColumn('idx')
        	Accumulate('hasdiabetes')
        ) as sqlmr

This gives the required flexibility to create sql templates flexible enough to be retrained and evaluated with different datasets and configuration. 

One nice add on we provide is the ability to ignore DROP TABLE errors. This works around the Teradata SQL issue of not supporting DROP TABLE IF EXISTS. 

To know what values are passed in the varargs, see [Additional Variables](#additional-variables).


## Additional Variables

The following additional variables are passed in the varargs support for each language 

| Name   |      Description      |
|----------|:-------------:|
| model_version |  The model version (trainedModel.id) for this model execution. |
| model_id | The model id of the model definition being executed. |
| model_table | For MLE models. It is based on the first part of the model version id and we prepend the string `AOA_MODELS_` to make it clear the table is a model version managed by the AOA. An example table name is `AOA_MODELS_ba80ac47` where the model version is `ba80ac47-f868-449d-93e8-abf601fce8aa`. Evaluations and subsequent scoring in the MLE can use this table name then as it is associated with the trained model version.|


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


# CLI tools

## Install AOA Python Package

### Install from Python Package Index

```console
# pip install aoa
```

### Install from Source Code

```console
# git clone https://github.com/ThinkBigAnalytics/AoaPythonClient.git
# cd AoaPythonClient
# pip install .
```

## Running Models Locally

To aid developing and testing the models setup in the AOA locally and in the datalab, we provide some useful cli tools to 
run the model training and evaluation using the config and data that you expect to be passed during automation.

### Train

```console
# aoa --run --model_id <your-model-uuid> --mode train
Starting training...
Finished training
Saved trained model
```

### Evaluate

```console
# aoa --run --model_id <your-model-uuid> --mode evaluate
model accuracy is  0.96
Evaluation complete...
```

Note that R and SQL models are also launched using the python cli. That is because the python cli will prompt the user for required details and then execute R or SQL logic.


## Adding Models based on Templates

You can add models based using a cli tool based on model templates defined in the [model_templates](./model_templates) folder. 

```console
# aoa --add
Model Name: My first model
Model Description: My first model using AnalyticOpsAccelerator
These languages are supported:  python, R
Model Language: python
These templates are available for python:  empty, sklearn
Template type (leave blank for the empty one):
INFO:root:Creating model structure for model: <your-model-uuid>
```