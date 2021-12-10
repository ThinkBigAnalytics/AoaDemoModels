

# Models

This repository contains the example / demo models for Teradata AnalyticOps (ModelOps). The goal of these examples is to provide a simple reference implementation for users and not to provide a detailed data science example for each use case. We provide both Python and R examples along with Classification and Regression model types.

Refer to the [OfficialDocumentation](https://docs.tdaoa.com) for more information.

## Available Models

## Categorical

- [Python (Diabetes)](model_definitions/03c9a01f-bd46-4e7c-9a60-4282039094e6)
- [R (Diabetes)](model_definitions/bf6a52b2-b595-4358-ac4f-24fb41a85c45)
- [Python Partitioned Modelling](model_definitions/dfd4052e-f91b-4aa5-9c79-f26d649dd931)

## Regression

- [Python (Demand Forecasting)](model_definitions/b1e18b12-5ccd-4c94-b96b-c2cebf230150)
- [VAL (Demand Forecasting)](model_definitions/db62c97c-2cd5-4f4e-af3c-b967c13d0327)


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
