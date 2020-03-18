- [Overview](#overview)
- [Assumptions](#assumptions)
- [Walkthrough](#walkthrough)
  - [0. Prototype script](#0-prototype-script)
  - [1. Add new model](#1-add-new-model)
  - [2. Create python environment and add packages](#2-create-python-environment-and-add-packages)
  - [3. Define and test training method](#3-define-and-test-training-method)
  - [4. Define and test evaluation method](#4-define-and-test-evaluation-method)
  - [5. Moving on to AOA UI](#5-moving-on-to-aoa-ui)


# Overview

This example shows step-by-step scenario to add a simple model to AOA service. We are using Pandas dataframes here, so it could be easily adopted to use Teradata or Spark dataframes.

# Assumptions

- aoa package is already installed

# Walkthrough

This walkthrough illustrates adding new model to framework, starting with local development and moving to the service for training and evaluation

## 0. Prototype script

We have the following propotype script that implements simplest flower classifier.
```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
import io
import requests

# load data & engineer
response = requests.get('https://datahub.io/machine-learning/iris/r/iris.csv')
file_object = io.StringIO(response.content.decode('utf-8'))
df = pd.read_csv(file_object)
# split dataset
train_df, predict_df = train_test_split(df, test_size = 0.5) 
features = 'sepallength,sepalwidth,petallength,petalwidth'.split(',')
X_train = train_df.loc[:, features]
y_train = train_df['class']

print("Starting training...")
# fit model to training data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
print("Finished training")

# evaluate model against test dataset
X_predict = predict_df.loc[:, features]
y_test = predict_df['class']
y_predict = knn.predict(X_predict)
print("model accuracy is ", metrics.accuracy_score(y_test, y_predict))

# save model
joblib.dump(knn, 'iris_knn.joblib')
print("Saved trained model")
```

We are starting in model root directory.

## 1. Add new model

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

## 2. Create python environment and add packages

```console
# python3 -m venv venv
# source venv/bin/activate
# cat > model_modules/requirements.txt <<EOF
numpy==1.16.1
pandas==0.24.2
python-dateutil==2.8.0
pytz==2019.1
scikit-learn==0.20.4
scipy==1.3.0
six==1.12.0
EOF
# pip3 install -r model_modules/requirements.txt
```

## 3. Define and test training method

Let's edit model_modules/training.py

```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import os
import io
import requests

def train(data_conf, model_conf, **kwargs):
    """Python train method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """

    hyperparams = model_conf["hyperParameters"]

    # load data & engineer
    response = requests.get(data_conf['location'])
    file_object = io.StringIO(response.content.decode('utf-8'))
    train_df = pd.read_csv(file_object)
    features = 'sepallength,sepalwidth,petallength,petalwidth'.split(',')
    X = train_df.loc[:, features]
    y = train_df['class']

    print("Starting training...")
    # fit model to training data
    knn = KNeighborsClassifier(n_neighbors=hyperparams['n_neighbors'])
    knn.fit(X,y)
    print("Finished training")

    # export model artefacts to models/ folder
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(knn, 'models/iris_knn.joblib')
    print("Saved trained model")
```
Notice the parameters:
- `data_conf['location']` used to provide path to data
- `hyperparams['n_neighbors']` used to provide classifier config

Let's add model parameters to config.json:
```json
{
  "hyperParameters": {
    "n_neighbors": 3
  }
}
```
We shall test this method with local CLI client, but we need some data first, so we need to create data configuration file `model_definitions/<your-model-uuid>/.cli/datasets/train.json`. 

*Note that you can use the cli without specifying modelid, mode and dataset and it will prompt you to select between the available options.*

```console
# mkdir -p model_definitions/<your-model-uuid>/.cli/datasets
# cat > model_definitions/<your-model-uuid>/.cli/datasets/train.json <<EOF
{
  "location": "https://datahub.io/machine-learning/iris/r/iris.csv"
}
EOF
```
Train model locally
```console
# aoa --run --model_id <your-model-uuid> --mode train
Starting training...
Finished training
Saved trained model
```
## 4. Define and test evaluation method

Let's edit model_modules/scoring.py

```python
import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
import json
import io
import requests

def evaluate(data_conf, model_conf, **kwargs):
    """Python evaluate method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """
    response = requests.get(data_conf['location'])
    file_object = io.StringIO(response.content.decode('utf-8'))
    predict_df = pd.read_csv(file_object)
    features = 'sepallength,sepalwidth,petallength,petalwidth'.split(',')
    X_predict = predict_df.loc[:, features]
    y_test = predict_df['class']
    knn = joblib.load('models/iris_knn.joblib')

    y_predict = knn.predict(X_predict)
    scores = {}
    scores['accuracy'] = metrics.accuracy_score(y_test, y_predict)
    print("model accuracy is ", scores['accuracy'])

    # dump results as json file evaluation.json to models/ folder
    with open("models/evaluation.json", "w+") as f:
        json.dump(scores, f)
    print("Evaluation complete...")
```
In order to evaluate this model we would need evaluation dataset, but we will use the same dataset at the moment:
```console
# aoa --run --model_id <your-model-uuid> --mode evaluate
model accuracy is  0.96
Evaluation complete...
```
## 5. Moving on to AOA UI

So we have our model training and evaluation tested locally, the next step is to train the model in AOA UI.