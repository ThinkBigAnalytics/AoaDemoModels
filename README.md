

# Models

This repository contains demo models for the Teradata AOA. The goal of these demo models is to provide a reference for implementing models with the AOA and also to show the full set of capabilities. To make the examples more consistent, we choose the same simple classification problem for each of them.


If you want to see more details around how to use the AOA, please watch the following video series

- [Project Setup](https://web.microsoftstream.com/video/ed11c0ef-2292-476f-aff2-723d7207676e)
- [Model Training](https://web.microsoftstream.com/video/20062b89-1296-4249-a5bd-8e1d62c91c23)
- [Model Evaluation](https://web.microsoftstream.com/video/db169a39-d8c6-4e1e-996b-fa63a7449b86)
- [Model Deployment](https://web.microsoftstream.com/video/f027aa32-3403-433a-8f6b-0fb5074d498e)

# Available Models

## Categorical

- [Python XGboost (Diabetes)](model_definitions/03c9a01f-bd46-4e7c-9a60-4282039094e6)
- [R GBM (Diabetes)](model_definitions/bf6a52b2-b595-4358-ac4f-24fb41a85c45)
- [Python Micro-Model](model_definitions/dfd4052e-f91b-4aa5-9c79-f26d649dd931)
- [PySpark XGboost (Diabetes)](model_definitions/149e31ed-c554-46b4-95d2-00c5c43320fb)

## Regression

- [Python RF Regressor (Demand Forecasting)](model_definitions/b1e18b12-5ccd-4c94-b96b-c2cebf230150)

# Adding new Models

Use the aoa cli (see [here](https://pypi.org/project/aoa/))

```
aoa add
```

## Running Models Locally

Use the aoa cli (see [here](https://pypi.org/project/aoa/))

```
aoa run -h
```
