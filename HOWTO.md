# Overview
This example shows step-by-step scenario to add a simple model to AOA service. We are using Pandas dataframes here, so it could be easily adopted to use Teradata or Spark dataframes.
# Datasets
We would need training and evaluation datasets, you could find some examples in `model_modules/data` directory.
# Walkthrough
This walkthrough illustrates adding new model to framework, starting with local development and moving to the service for training and evaluation
## 0. Prototype script
We have the following propotype script that implements simplest flower classifier.
```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn import metrics

# load data & engineer
train_df = pd.read_csv('iris_train1.csv')
features = 'SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm'.split(',')
X_train = train_df.loc[:, features]
y_train = train_df.Species

print("Starting training...")
# fit model to training data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
print("Finished training")

# evaluate model against test dataset
predict_df = pd.read_csv('iris_evaluate.csv')
X_predict = predict_df.loc[:, features]
y_test = predict_df.Species
y_predict = knn.predict(X_predict)
print("model accuracy is ", metrics.accuracy_score(y_test, y_predict))

# save model
joblib.dump(knn, 'iris_knn.joblib')
print("Saved trained model")
```
We are starting in AoaDemoModels root directory.
## 1. Add new model
```console
# ./cli/repo-cli.py -a
Model Name: My first model
Model Description: My first model using AnalyticOpsAccelerator
These languages are supported:  python, R
Model Language: python
These templates are available for python:  empty, sklearn
Template type (leave blank for the empty one):
INFO:root:Creating model structure for model: <your-model-uuid>
# cd ./model_definitions/<your-model-uuid>/DOCKER
```
## 2. Create python environment and add some packages
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
    train_df = pd.read_csv(data_conf['data_table'])
    features = 'SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm'.split(',')
    X = train_df.loc[:, features]
    y = train_df.Species

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
- `data_conf['data_table']` used to provide path to data
- `hyperparams['n_neighbors']` used to provide classifier config

Let's add model parameters to config.json:
```json
{
  "hyperParameters": {
    "n_neighbors": 3
  }
}
```
We shall test this method with local CLI client, but we need some data first:
- Make sure your training dataset is available in `model_modules/data/iris_train1.csv`
- define data configuration file `examples/train_dataset1.json`
```console
# cat > examples/train_dataset1.json <<EOF
{
  "data_table": "./model_modules/data/iris_train1.csv"
}
EOF
# ../../../cli/run-model-cli.py -d examples/train_dataset1.json <your-model-uuid> train
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

def evaluate(data_conf, model_conf, **kwargs):
    """Python evaluate method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """

    predict_df = pd.read_csv(data_conf['data_table'])
    features = 'SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm'.split(',')
    X_predict = predict_df.loc[:, features]
    y_test = predict_df.Species
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
In order to evaluate this model we would need evaluation dataset (make sure it exists in this path):
```console
# cat > examples/evaluate_dataset.json <<EOF
{
  "data_table": "./model_modules/data/iris_evaluate.csv"
}
EOF
# ../../../cli/run-model-cli.py -d examples/evaluate_dataset.json <your-model-uuid> evaluate
model accuracy is  0.96
Evaluation complete...
```
## 5. Moving on to AOA UI
So we have our model training and evaluation tested locally, the next step is to train the model in AOA UI.

Depending on your demo setup you either need to:
- commit and push the model to git; 
- or just make sure your demo AoaCore service runs against your models directory

In order to run demo training and evaluation you must include data inside model_modules/data directory in your commit.

To train and evaluate models you would need to register datasets, you could do that manually using metadata in train/evaluate_dataset json files, or you could use script `examples/register_datasets.sh`

Once datasets are registered feel free to train this model in API/UI against different datasets, evaluate and compare results.