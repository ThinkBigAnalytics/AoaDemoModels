# Overview

Simple sklearn RandomForestClassifier for Iris dataset to showcase few features of AnalyticOps framework:

- Store evaluation artifacts additional to metrics in metadata (images on S3)
- Expose explanation endpoint for RESTful deployment

# Datasets

The dataset required to train or eval this model is Iris dataset with the following descriptor:
```json
{
    "location": "https://aoa-public-datasets.s3-eu-west-1.amazonaws.com/AoaPmmlDemo/iris_full.csv"
}
```

# Training

The [training.py](model_modules/training.py) produces the following artifacts:

- model.joblib (sklearn pickle file with model)
- X_train.pickle (pandas pickle file with training set, needed to setup explainers during evaluation and scoring)

# Evaluation

Evaluation is defined in the `evaluate` method in [scoring.py](model_modules/scoring.py), it returns model accuracy and creates summary plot for SHAP values, which is stored as an evaluation artifact on S3

# Scoring

This demo supports RESTful deployment, it exposes to methods:

- `predict` returns predicted value, for example: `curl -X POST http://localhost:5000/predict -d '{"data": {"ndarray": [5.7, 3.0, 4.2, 1.2] }}'`
```json
{
  "data": {
    "ndarray": [
      "Iris-versicolor"
    ]
  }
}
```
- `explain` returns weights of features contributing to this prediction, for example: `curl -X POST http://localhost:5000/explain -d '{"data": {"ndarray": [5.7, 3.0, 4.2, 1.2] }}'`
```json
{
  "data": {
    "ndarray": {
      "versicolor": [
        [
          "petal length (cm)",
          0.30644806228509874
        ],
        [
          "petal width (cm)",
          0.2970124001480905
        ],
        [
          "sepal length (cm)",
          0.02873420859501652
        ],
        [
          "sepal width (cm)",
          -0.00013398101799354164
        ]
      ]
    }
  }
}
```