

# Models

This repository contains the example / demo models for Teradata AnalyticOps (ModelOps). The goal of these examples is to provide a simple reference implementation for users and not to provide a detailed data science example for each use case. We provide both Python and R examples along with Classification and Regression model types.

Refer to the [OfficialDocumentation](https://docs.tdaoa.com) for more information.

## Available Models

## Categorical

- [Python (Diabetes)](model_definitions/python-diabetes)
- [R (Diabetes)](model_definitions/r-diabetes)
- [Python Partitioned Modelling](model_definitions/python-partitioned-modelling)

## Regression

- [Python (Demand Forecasting)](model_definitions/python-demand-forecast)
- [VAL (Demand Forecasting)](model_definitions/python-val-forecast)


## Adding new Models

Use the aoa cli (see [here](https://pypi.org/project/aoa/))

```
aoa add
```

## Running Models Locally

Use the aoa cli (see [here](https://pypi.org/project/aoa/))

```
aoa run -h
```
